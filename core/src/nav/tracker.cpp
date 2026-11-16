#include "roversim/nav/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace roversim::nav {

void ObstacleTracker::update(const Pose2& sensor, const ScanGeometry& scan, std::span<const float> ranges,
                             std::span<const std::uint8_t> transient, double dt,
                             std::vector<std::uint8_t>& ray_flags) {
    const int n = static_cast<int>(std::min({ranges.size(), transient.size(), static_cast<std::size_t>(scan.num_rays)}));
    ray_flags.assign(ranges.size(), ray_flag::Static);
    detections_.clear();

    const auto range = [&](int i) { return static_cast<double>(ranges[static_cast<std::size_t>(i)]); };
    const auto candidate = [&](int i) { return transient[static_cast<std::size_t>(i)] != 0 && range(i) < cfg_.max_range; };
    // Скан на полный оборот замкнут: за последним лучом идёт первый.
    const bool circular = n > 1 && std::abs(scan.angle_increment * n - kTwoPi) < 1e-6;
    const auto connected = [&](int a, int b) {
        return candidate(a) && candidate(b) && std::abs(range(a) - range(b)) <= cfg_.range_jump;
    };

    // ---- 1. Сегментация лучей-кандидатов ----
    // Обход начинается с разрыва, чтобы объект на стыке конца и начала скана не
    // распался на два сегмента.
    int begin = 0;
    if (circular) {
        begin = -1;
        for (int i = 0; i < n; ++i) {
            if (!connected((i + n - 1) % n, i)) {
                begin = i;
                break;
            }
        }
    }
    if (begin >= 0) {
        const double c = std::cos(sensor.theta), s = std::sin(sensor.theta);
        int k = 0;
        while (k < n) {
            const int first = (begin + k) % n;
            if (!candidate(first)) {
                ++k;
                continue;
            }
            int length = 1;
            while (k + length < n && connected((begin + k + length - 1) % n, (begin + k + length) % n)) ++length;
            k += length;
            if (length < cfg_.min_points) continue;

            const auto point = [&](int j) {  // точка луча сегмента в СК сенсора
                const double a = scan.angle_min + (first + j) * scan.angle_increment;
                return Vec2{std::cos(a), std::sin(a)} * range((first + j) % n);
            };
            const double chord = distance(point(0), point(length - 1));
            if (chord > cfg_.max_size) continue;
            // Круг по видимой дуге: радиус — половина хорды, центр — за ближайшей
            // точкой на луче, проходящем через середину сегмента.
            double nearest = std::numeric_limits<double>::infinity();
            for (int j = 0; j < length; ++j) nearest = std::min(nearest, range((first + j) % n));
            const double radius = std::clamp(0.5 * chord, 0.05, 0.5 * cfg_.max_size);
            const double mid = scan.angle_min + (first + 0.5 * (length - 1)) * scan.angle_increment;
            const Vec2 center = Vec2{std::cos(mid), std::sin(mid)} * (nearest + radius);
            detections_.push_back({sensor.position() + rotate(center, c, s), radius, first, length, -1});
        }
    }

    // ---- 2. Прогноз треков и привязка измерений (жадно, к ближайшему) ----
    for (Track& track : tracks_) track.position += track.velocity * dt;
    used_.assign(tracks_.size(), 0);
    for (Detection& detection : detections_) {
        double best = cfg_.gate;
        for (std::size_t t = 0; t < tracks_.size(); ++t) {
            if (used_[t] != 0) continue;
            const double d = distance(detection.position, tracks_[t].position);
            if (d < best) {
                best = d;
                detection.track = static_cast<int>(t);
            }
        }
        if (detection.track >= 0) used_[static_cast<std::size_t>(detection.track)] = 1;
    }

    // ---- 3. Обновление: альфа-бета фильтр ----
    //   p <- p_прогноз + alpha * r,   v <- v + (beta / dt) * r,   r = измерение - p_прогноз.
    const std::size_t existing = tracks_.size();
    for (const Detection& detection : detections_) {
        std::uint8_t flag = ray_flag::Dynamic;  // лучи объекта в карту не идут
        if (detection.track < 0) {
            Track fresh;
            fresh.id = next_id_++;
            fresh.position = fresh.origin = fresh.anchor = detection.position;
            fresh.radius = detection.radius;
            fresh.hits = 1;
            tracks_.push_back(fresh);
        } else {
            Track& track = tracks_[static_cast<std::size_t>(detection.track)];
            const Vec2 residual = detection.position - track.position;
            track.position += residual * cfg_.alpha;
            if (dt > 0.0) track.velocity += residual * (cfg_.beta / dt);
            const double speed = norm(track.velocity);
            if (speed > cfg_.max_speed) track.velocity = track.velocity * (cfg_.max_speed / speed);
            track.radius = 0.7 * track.radius + 0.3 * detection.radius;
            ++track.hits;
            track.misses = 0;

            // «Парковка»: объект, который долго не уходит от опорной точки, перестаёт
            // считаться подвижным и отдаётся карте как обычное препятствие.
            if (distance(track.position, track.anchor) > cfg_.park_travel) {
                track.anchor = track.position;
                track.still_time = 0.0;
            } else {
                track.still_time += dt;
            }
            const bool parked = track.still_time > cfg_.park_time;
            // При confirm_travel > 0 подвижным объект считается не с момента
            // подтверждения, а когда он действительно сместился: неподвижный
            // предмет, проявившийся на «полу» (например, торец стены, который карта
            // успела стереть), тогда остаётся просто препятствием и через park_time
            // возвращается в карту.
            track.displaced = track.displaced || cfg_.confirm_travel <= 0.0 ||
                              distance(track.position, track.origin) > cfg_.confirm_travel;
            track.moving = track.hits >= cfg_.confirm_hits && track.displaced && !parked;
            if (parked) flag = ray_flag::Settled;
        }
        for (int j = 0; j < detection.length; ++j) {
            ray_flags[static_cast<std::size_t>((detection.first + j) % n)] = flag;
        }
    }
    for (std::size_t t = 0; t < existing; ++t) {
        if (used_[t] == 0) ++tracks_[t].misses;  // объект не виден: трек живёт на прогнозе
    }
    std::erase_if(tracks_, [&](const Track& track) { return track.misses > cfg_.max_misses; });
}

}  // namespace roversim::nav
