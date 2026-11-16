#include "roversim/nav/slam.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace roversim::nav {

GridSlam::GridSlam(const SlamConfig& config, const ScanGeometry& scan, const Pose2& initial_pose)
    : cfg_(config), scan_(scan), map_(config.map, initial_pose.position()), pose_(initial_pose) {
    if (scan_.num_rays < 1) throw std::invalid_argument("ScanGeometry: num_rays must be >= 1");
    if (cfg_.match_points < 1) throw std::invalid_argument("SlamConfig: match_points must be >= 1");
    dirs_.reserve(static_cast<std::size_t>(scan_.num_rays));
    for (int i = 0; i < scan_.num_rays; ++i) {
        const double a = scan_.angle_min + i * scan_.angle_increment;
        dirs_.push_back({std::cos(a), std::sin(a)});
    }
}

Pose2 GridSlam::sensor_pose() const noexcept {
    const double c = std::cos(pose_.theta), s = std::sin(pose_.theta);
    const Vec2 p = pose_.position() + rotate({scan_.mount.x, scan_.mount.y}, c, s);
    return {p.x, p.y, pose_.theta + scan_.mount.theta};
}

const Pose2& GridSlam::update(const Pose2& odom_pose, std::span<const float> ranges) {
    localize(odom_pose, ranges);
    integrate(ranges);
    return pose_;
}

const Pose2& GridSlam::localize(const Pose2& odom_pose, std::span<const float> ranges) {
    // 1. Прогноз: относительное перемещение по одометрии, выраженное в прошлой
    //    СК робота, применяется к прошлой оценке позы. Дрейф самой одометрии при
    //    этом не важен — используется только её приращение за один шаг.
    Pose2 predicted = pose_;
    if (has_odom_) {
        const double c0 = std::cos(last_odom_.theta), s0 = std::sin(last_odom_.theta);
        const Vec2 local = rotate_inv(odom_pose.position() - last_odom_.position(), c0, s0);
        const Vec2 moved = rotate(local, std::cos(pose_.theta), std::sin(pose_.theta));
        predicted = {pose_.x + moved.x, pose_.y + moved.y,
                     wrap_angle(pose_.theta + wrap_angle(odom_pose.theta - last_odom_.theta))};
    }
    last_odom_ = odom_pose;
    has_odom_ = true;

    // 2. Точки скана в СК робота. Берётся равномерная выборка лучей с отражением.
    points_.clear();
    const std::size_t n = std::min(ranges.size(), dirs_.size());
    const std::size_t stride = std::max<std::size_t>(1, n / static_cast<std::size_t>(cfg_.match_points));
    const double cm = std::cos(scan_.mount.theta), sm = std::sin(scan_.mount.theta);
    const double no_return = scan_.range_max - 1e-3;
    for (std::size_t i = 0; i < n; i += stride) {
        if (ranges[i] >= no_return) continue;
        const Vec2 in_sensor = dirs_[i] * static_cast<double>(ranges[i]);
        points_.push_back(Vec2{scan_.mount.x, scan_.mount.y} + rotate(in_sensor, cm, sm));
    }

    // 3. Коррекция по карте.
    matched_ = false;
    score_ = 0.0;
    pose_ = predicted;
    if (cfg_.scan_matching && static_cast<int>(map_.scans()) >= cfg_.warmup_scans && points_.size() >= 10) {
        const Pose2 refined = match(predicted);
        if (score_ >= cfg_.min_score) {
            pose_ = refined;
            matched_ = true;
        }
    }
    return pose_;
}

void GridSlam::integrate(std::span<const float> ranges, std::span<const std::uint8_t> ray_flags) {
    map_.integrate_scan(sensor_pose(), dirs_, ranges, scan_.range_max, ray_flags);
}

double GridSlam::map_score(const Pose2& robot) const noexcept {
    const double c = std::cos(robot.theta), s = std::sin(robot.theta);
    double sum = 0.0;
    for (const Vec2& p : points_) sum += map_.occupancy_value(robot.position() + rotate(p, c, s));
    return sum / static_cast<double>(points_.size());
}

// Сопоставление скана с картой. Максимизируется
//     J(x) = (1/N) * sum_i M(T(x) p_i)  -  k_t * |t - t0|^2  -  k_r * (theta - theta0)^2,
// где M — гладкое (билинейное) значение карты под точкой скана, x = (t, theta) —
// поза, x0 — прогноз по одометрии. Первый член велик, когда точки скана лежат на
// занятых клетках, и отрицателен, если они попадают в заведомо свободное место;
// штраф — априорное распределение вокруг прогноза.
//
// Поиск — покоординатный подъём с дроблением шага: из текущей позы пробуются
// шесть соседних (по ±шаг вдоль x, y и по углу), выбирается лучшая; когда
// улучшения нет, шаг делится пополам. Производные не нужны, а разрешение после
// четырёх уровней — миллиметры и сотые доли градуса.
Pose2 GridSlam::match(const Pose2& predicted) {
    const auto objective = [&](const Pose2& pose) {
        const double dx = pose.x - predicted.x, dy = pose.y - predicted.y;
        const double da = wrap_angle(pose.theta - predicted.theta);
        return map_score(pose) - cfg_.prior_linear * (dx * dx + dy * dy) - cfg_.prior_angular * da * da;
    };

    Pose2 best = predicted;
    double best_value = objective(best);
    double linear = cfg_.linear_step, angular = cfg_.angular_step;
    for (int level = 0; level < cfg_.refinements; ++level) {
        for (int iteration = 0; iteration < cfg_.max_iterations; ++iteration) {
            const Pose2 candidates[6] = {
                {best.x + linear, best.y, best.theta}, {best.x - linear, best.y, best.theta},
                {best.x, best.y + linear, best.theta}, {best.x, best.y - linear, best.theta},
                {best.x, best.y, best.theta + angular}, {best.x, best.y, best.theta - angular},
            };
            int winner = -1;
            double winner_value = best_value;
            for (int k = 0; k < 6; ++k) {
                const double value = objective(candidates[k]);
                if (value > winner_value + 1e-9) {
                    winner_value = value;
                    winner = k;
                }
            }
            if (winner < 0) break;
            best = candidates[winner];
            best_value = winner_value;
        }
        linear *= 0.5;
        angular *= 0.5;
    }
    best.theta = wrap_angle(best.theta);
    score_ = map_score(best);
    return best;
}

}  // namespace roversim::nav
