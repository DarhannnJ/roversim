#include "roversim/nav/dwa.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "roversim/nav/planner.hpp"

namespace roversim::nav {

DwaPlanner::DwaPlanner(const DwaConfig& config) : cfg_(config) {
    if (cfg_.speed_samples < 2 || cfg_.yaw_samples < 3) throw std::invalid_argument("DwaConfig: too few velocity samples");
    if (!(cfg_.time_step > 0.0) || !(cfg_.horizon >= cfg_.time_step)) throw std::invalid_argument("DwaConfig: bad horizon or time_step");
    if (!(cfg_.max_accel > 0.0) || !(cfg_.max_yaw_accel > 0.0) || !(cfg_.max_speed > 0.0) || !(cfg_.max_yaw_rate > 0.0)) {
        throw std::invalid_argument("DwaConfig: speed and acceleration limits must be positive");
    }
    if (!(cfg_.grid_resolution > 0.0) || !(cfg_.grid_half_size > 0.0)) throw std::invalid_argument("DwaConfig: bad local grid");
    grid_size_ = 2 * static_cast<int>(std::ceil(cfg_.grid_half_size / cfg_.grid_resolution));
    mask_.assign(static_cast<std::size_t>(grid_size_) * grid_size_, 0);
}

// Расстояние от точки до ближайшей статической точки препятствия [м]: билинейная
// интерполяция локального поля расстояний. Одно чтение таблицы вместо перебора
// всех точек скана для каждой точки каждой траектории.
double DwaPlanner::static_distance(Vec2 p) const noexcept {
    constexpr double kFar = 1e3;
    if (grid_empty_) return kFar;
    const double gx = (p.x + cfg_.grid_half_size) / cfg_.grid_resolution - 0.5;
    const double gy = (p.y + cfg_.grid_half_size) / cfg_.grid_resolution - 0.5;
    const double fx0 = std::floor(gx), fy0 = std::floor(gy);
    const int ix = static_cast<int>(fx0), iy = static_cast<int>(fy0);
    if (ix < 0 || iy < 0 || ix + 1 >= grid_size_ || iy + 1 >= grid_size_) return kFar;  // вне сетки препятствий не видно
    const double tx = gx - fx0, ty = gy - fy0;
    const float* row0 = &field_[static_cast<std::size_t>(iy) * grid_size_ + ix];
    const float* row1 = row0 + grid_size_;
    const double bottom = row0[0] + (row0[1] - row0[0]) * tx;
    const double top = row1[0] + (row1[1] - row1[0]) * tx;
    return (bottom + (top - bottom) * ty) * cfg_.grid_resolution;
}

DwaResult DwaPlanner::compute(std::span<const Vec2> obstacles, std::span<const MovingObstacle> moving, Vec2 target,
                              double v, double w, bool allow_reverse) {
    // ---- 1. Локальное поле расстояний по точкам скана ----
    // Точки скана, лежащие на сопровождаемом подвижном объекте, в статическую
    // картину не идут: объект уже учтён своей моделью — с прогнозом движения и
    // запасом moving_margin, который закрывает как раз эту область. Статические и
    // подвижные препятствия проверяются раздельно (см. пороги ниже).
    std::fill(mask_.begin(), mask_.end(), std::uint8_t{0});
    grid_empty_ = true;
    for (const Vec2& p : obstacles) {
        bool on_moving_object = false;
        for (const MovingObstacle& m : moving) {
            const double reach = m.radius + cfg_.moving_margin;
            const Vec2 d = p - m.position;
            on_moving_object = on_moving_object || dot(d, d) < reach * reach;
        }
        if (on_moving_object) continue;
        const int ix = static_cast<int>(std::floor((p.x + cfg_.grid_half_size) / cfg_.grid_resolution));
        const int iy = static_cast<int>(std::floor((p.y + cfg_.grid_half_size) / cfg_.grid_resolution));
        if (ix < 0 || iy < 0 || ix >= grid_size_ || iy >= grid_size_) continue;
        mask_[static_cast<std::size_t>(iy) * grid_size_ + ix] = 1;
        grid_empty_ = false;
    }
    if (!grid_empty_) distance_transform(mask_, grid_size_, grid_size_, field_);

    // ---- 2. Динамическое окно ----
    // Пересечение допустимых скоростей робота с теми, что достижимы из текущих
    // (v, w) за время window_time при ограниченных ускорениях.
    const double min_speed = allow_reverse ? std::min(cfg_.min_speed, cfg_.escape_speed) : cfg_.min_speed;
    v = std::clamp(v, min_speed, cfg_.max_speed);
    w = std::clamp(w, -cfg_.max_yaw_rate, cfg_.max_yaw_rate);
    const double v_lo = std::max(min_speed, v - cfg_.max_accel * cfg_.window_time);
    const double v_hi = std::min(cfg_.max_speed, v + cfg_.max_accel * cfg_.window_time);
    const double w_lo = std::max(-cfg_.max_yaw_rate, w - cfg_.max_yaw_accel * cfg_.window_time);
    const double w_hi = std::min(cfg_.max_yaw_rate, w + cfg_.max_yaw_accel * cfg_.window_time);

    // Зазор до статических препятствий и до подвижных (в их прогнозируемом
    // положении на момент t) считается раздельно.
    const auto static_clearance = [&](Vec2 p) { return static_distance(p) - cfg_.robot_radius; };
    const auto moving_clearance = [&](Vec2 p, double t) {
        double clear = std::numeric_limits<double>::infinity();
        for (const MovingObstacle& m : moving) {
            clear = std::min(clear, distance(p, m.position + m.velocity * t) - m.radius - cfg_.robot_radius - cfg_.moving_margin);
        }
        return clear;
    };
    // Обычно траектория обязана выдерживать зазор safety_margin. Если робот уже
    // оказался ближе, требовать этого бессмысленно (не осталось бы ни одной
    // траектории), и порогом становится текущий зазор: стоять и поворачиваться
    // можно, удаляться можно, приближаться нельзя ни на сколько. Порог именно
    // равен текущему зазору, без послабления: иначе робот такт за тактом
    // «подползал» бы к стене.
    //
    // Порогов два — для стен и для подвижных объектов. С общим порогом робот,
    // которого коснулось подвижное препятствие, терял бы запас и до стен: порог
    // падал бы до нуля для всего сразу, и, отъезжая от препятствия, робот мог
    // упереться в стену.
    const double static_start = static_clearance({0.0, 0.0});
    const double moving_start = moving_clearance({0.0, 0.0}, 0.0);
    const double static_threshold = std::min(cfg_.safety_margin, static_start);
    const double moving_threshold = std::min(cfg_.safety_margin, moving_start);

    struct Candidate {
        double v, w, goal_distance, heading_error, clearance, survival;
        std::size_t trajectory_begin, trajectory_end;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(cfg_.speed_samples) * cfg_.yaw_samples);
    trajectory_.clear();
    const int steps = std::max(1, static_cast<int>(std::lround(cfg_.horizon / cfg_.time_step)));
    const double dt = cfg_.time_step;

    for (int i = 0; i < cfg_.speed_samples; ++i) {
        double vi = v_lo + (v_hi - v_lo) * i / (cfg_.speed_samples - 1);
        // Нулевая скорость (поворот на месте) должна быть среди вариантов, если она в окне.
        if (v_lo < 0.0 && v_hi > 0.0 && std::abs(vi) < 0.5 * (v_hi - v_lo) / (cfg_.speed_samples - 1)) vi = 0.0;
        for (int j = 0; j < cfg_.yaw_samples; ++j) {
            double wj = w_lo + (w_hi - w_lo) * j / (cfg_.yaw_samples - 1);
            if (w_lo < 0.0 && w_hi > 0.0 && std::abs(wj) < 0.5 * (w_hi - w_lo) / (cfg_.yaw_samples - 1)) wj = 0.0;

            // Перемещение за один шаг в СК начала шага — дуга окружности (точно, как в симуляторе).
            const double phi = wj * dt;
            double sinc = 1.0, cosc = 0.0;
            if (std::abs(phi) > 1e-6) {
                sinc = std::sin(phi) / phi;
                cosc = (1.0 - std::cos(phi)) / phi;
            }
            const Vec2 local_step{vi * dt * sinc, vi * dt * cosc};
            const double cos_phi = std::cos(phi), sin_phi = std::sin(phi);

            // ---- 3. Моделирование траектории ----
            Vec2 position{0.0, 0.0}, safe_position{0.0, 0.0};
            double cos_h = 1.0, sin_h = 0.0, safe_heading = 0.0;
            double min_clearance = std::numeric_limits<double>::infinity();
            int collision_step = 0;  // 0 — столкновений на горизонте нет
            const std::size_t begin = trajectory_.size();
            for (int k = 1; k <= steps; ++k) {
                position += rotate(local_step, cos_h, sin_h);
                const double next_cos = cos_h * cos_phi - sin_h * sin_phi;
                sin_h = sin_h * cos_phi + cos_h * sin_phi;
                cos_h = next_cos;
                const double clear_static = static_clearance(position);
                const double clear_moving = moving_clearance(position, k * dt);
                if (clear_static < static_threshold || clear_moving < moving_threshold) {
                    collision_step = k;
                    break;
                }
                min_clearance = std::min({min_clearance, clear_static, clear_moving});
                safe_position = position;
                safe_heading = phi * k;
                trajectory_.push_back(position);
            }

            // Условие допустимости: тормозной путь v²/(2a) умещается в участок
            // траектории до столкновения. Поворот на месте допустим всегда.
            if (collision_step > 0) {
                const double free_path = std::abs(vi) * (collision_step - 1) * dt;
                const double braking = vi * vi / (2.0 * cfg_.max_accel);
                if (std::abs(vi) > 1e-9 && braking > free_path) {
                    trajectory_.resize(begin);
                    continue;
                }
            }
            if (!std::isfinite(min_clearance)) min_clearance = std::min(static_start, moving_start);

            const Vec2 to_target = target - safe_position;
            const double heading_error = std::abs(wrap_angle(std::atan2(to_target.y, to_target.x) - safe_heading));
            const double survival = collision_step > 0 ? static_cast<double>(collision_step - 1) / steps : 1.0;
            candidates.push_back({vi, wj, norm(to_target), heading_error, std::min(min_clearance, cfg_.clearance_cap),
                                  survival, begin, trajectory_.size()});
        }
    }

    DwaResult result;
    result.admissible = static_cast<int>(candidates.size());
    best_trajectory_.clear();
    if (candidates.empty()) return result;

    // ---- 4. Целевая функция ----
    // Расстояние до цели нормируется по множеству допустимых траекторий (как в
    // исходной работе): так вес слагаемого не зависит от того, далеко ли цель.
    double nearest = std::numeric_limits<double>::infinity(), farthest = 0.0;
    for (const Candidate& c : candidates) {
        nearest = std::min(nearest, c.goal_distance);
        farthest = std::max(farthest, c.goal_distance);
    }
    const double spread = std::max(farthest - nearest, 1e-6);
    const Candidate* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const Candidate& c : candidates) {
        const double score = cfg_.weight_goal * (farthest - c.goal_distance) / spread +
                             cfg_.weight_heading * (1.0 - c.heading_error / kPi) +
                             cfg_.weight_clearance * std::max(c.clearance, 0.0) / cfg_.clearance_cap +
                             cfg_.weight_speed * std::max(c.v, 0.0) / cfg_.max_speed +
                             cfg_.weight_survival * c.survival;
        if (score > best_score) {
            best_score = score;
            best = &c;
        }
    }
    result.v = best->v;
    result.w = best->w;
    result.feasible = true;
    result.clearance = best->clearance;
    result.score = best_score;
    best_trajectory_.assign(trajectory_.begin() + static_cast<std::ptrdiff_t>(best->trajectory_begin),
                            trajectory_.begin() + static_cast<std::ptrdiff_t>(best->trajectory_end));
    return result;
}

}  // namespace roversim::nav
