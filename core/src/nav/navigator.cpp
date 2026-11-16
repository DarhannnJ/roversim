#include "roversim/nav/navigator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace roversim::nav {

ScanGeometry scan_geometry(const Lidar2D& lidar) {
    ScanGeometry scan;
    scan.num_rays = lidar.num_rays();
    scan.angle_min = lidar.angle_min();
    scan.angle_increment = lidar.angle_increment();
    scan.range_max = lidar.config().range_max;
    scan.mount = lidar.config().mount;
    return scan;
}

NavigatorConfig navigator_config(const RobotParams& robot, double control_dt) {
    NavigatorConfig config;
    config.control_dt = control_dt;
    config.wheel_radius = robot.wheel_radius;
    config.track_width = robot.track_width;
    config.max_wheel_speed = robot.max_wheel_speed;
    config.planner.robot_radius = robot.radius;
    config.planner.soft_radius = std::max(config.planner.soft_radius, robot.radius + config.planner.safety_margin + 0.2);
    config.dwa.robot_radius = robot.radius;
    // Крейсерская скорость — 80 % предельной: остаётся запас на управление поворотом.
    const double v_max = robot.wheel_radius * robot.max_wheel_speed;
    config.dwa.max_speed = 0.8 * v_max;
    config.dwa.max_yaw_rate = std::min(config.dwa.max_yaw_rate, 2.0 * v_max / robot.track_width);
    return config;
}

Navigator::Navigator(const NavigatorConfig& config, const ScanGeometry& scan, const Pose2& initial_pose)
    : cfg_(config),
      slam_(config.slam, scan, initial_pose),
      planner_(config.planner),
      dwa_(config.dwa),
      tracker_(config.tracker),
      progress_anchor_(initial_pose.position()) {
    const double cm = std::cos(scan.mount.theta), sm = std::sin(scan.mount.theta);
    ray_dirs_robot_.reserve(slam_.ray_directions().size());
    for (const Vec2& dir : slam_.ray_directions()) ray_dirs_robot_.push_back(rotate(dir, cm, sm));
}

void Navigator::set_goal(Vec2 goal) {
    goal_ = goal;
    has_goal_ = true;
    status_ = NavStatus::Driving;
    path_.clear();
    path_cursor_ = 0;
    next_plan_time_ = time_;  // маршрут построится на ближайшем такте
    progress_anchor_ = slam_.pose().position();
    progress_time_ = time_;
}

WheelCommand Navigator::wheels(double v, double w) const noexcept {
    // Обратная кинематика; при насыщении обе уставки уменьшаются одинаково.
    double left = (v - 0.5 * w * cfg_.track_width) / cfg_.wheel_radius;
    double right = (v + 0.5 * w * cfg_.track_width) / cfg_.wheel_radius;
    const double peak = std::max(std::abs(left), std::abs(right));
    if (peak > cfg_.max_wheel_speed) {
        left *= cfg_.max_wheel_speed / peak;
        right *= cfg_.max_wheel_speed / peak;
    }
    return {left, right};
}

// Проверка маршрута по свежей карте: не появилось ли на нём препятствие.
// Маршрут прокладывается и через неразведанные клетки, поэтому новая стена на
// пути — обычное дело; заметив её, навигатор перепланирует сразу, не дожидаясь
// таймера. Ближайшие 0.3 м не проверяются: там от препятствий уводит DWA.
bool Navigator::path_blocked() const {
    if (path_.size() < 2) return false;
    const OccupancyGrid& map = slam_.map();
    const double res = map.resolution();
    // Тот же критерий, что у планировщика (расстояние до границы занятой клетки
    // меньше радиуса с запасом), но на полклетки мягче — иначе маршрут, идущий
    // вплотную к запретной зоне, считался бы перекрытым и перестраивался зря.
    const double limit = (cfg_.planner.robot_radius + cfg_.planner.safety_margin) / res - 0.5;
    const int reach = static_cast<int>(std::ceil(limit));
    const double limit2 = limit * limit;
    double along = 0.0;
    for (std::size_t i = path_cursor_; i + 1 < path_.size() && along < cfg_.path_check_distance; ++i) {
        const Vec2 a = path_[i], b = path_[i + 1];
        const double length = distance(a, b);
        const int steps = std::max(1, static_cast<int>(std::ceil(length / (2.0 * res))));
        for (int k = 0; k <= steps; ++k) {
            const double s = length * k / steps;
            if (along + s < 0.3) continue;
            if (along + s > cfg_.path_check_distance) break;
            int cx = 0, cy = 0;
            if (!map.cell_of(a + (b - a) * (static_cast<double>(k) / steps), cx, cy)) continue;
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    if (dx * dx + dy * dy <= limit2 && map.state(cx + dx, cy + dy) == CellState::Occupied) return true;
                }
            }
        }
        along += length;
    }
    return false;
}

void Navigator::replan() {
    planner_.set_map(slam_.map(), slam_.pose().position(), goal_);
    std::vector<Vec2> fresh;
    if (planner_.plan(slam_.pose().position(), goal_, fresh)) {
        path_.swap(fresh);
        path_cursor_ = 0;
        if (status_ == NavStatus::NoPath) status_ = NavStatus::Driving;
    } else {
        path_.clear();
        status_ = NavStatus::NoPath;
    }
    ++replans_;
    last_plan_time_ = time_;
    next_plan_time_ = time_ + cfg_.replan_period;
}

// Локальная цель («морковка»): точка маршрута на расстоянии lookahead впереди
// проекции робота на маршрут. Поиск проекции идёт вперёд от прошлого положения:
// робот не «перескакивает» на соседнюю ветку маршрута, проходящую рядом.
Vec2 Navigator::lookahead_point(Vec2 position) {
    if (path_.size() < 2) return goal_;
    std::size_t segment = path_cursor_;
    double best = std::numeric_limits<double>::infinity(), best_t = 0.0;
    for (std::size_t i = path_cursor_; i + 1 < path_.size(); ++i) {
        const Vec2 a = path_[i], ab = path_[i + 1] - a;
        const double length2 = dot(ab, ab);
        const double t = length2 > 0.0 ? std::clamp(dot(position - a, ab) / length2, 0.0, 1.0) : 0.0;
        const double d2 = dot(position - (a + ab * t), position - (a + ab * t));
        if (d2 < best) {
            best = d2;
            segment = i;
            best_t = t;
        }
    }
    path_cursor_ = segment;

    double remaining = cfg_.lookahead;
    Vec2 point = path_[segment] + (path_[segment + 1] - path_[segment]) * best_t;
    for (std::size_t i = segment; i + 1 < path_.size(); ++i) {
        const double d = distance(point, path_[i + 1]);
        if (d >= remaining) return point + (path_[i + 1] - point) * (remaining / d);
        remaining -= d;
        point = path_[i + 1];
    }
    return path_.back();
}

WheelCommand Navigator::update(std::span<const float> ranges, const Pose2& odom, double odom_v, double odom_w) {
    time_ += cfg_.control_dt;

    // ---- 1. Восприятие: поза → подвижные объекты → карта ----
    // Порядок важен: трекеру нужна уточнённая поза, а карте — знание о том,
    // какие лучи попали в подвижные объекты.
    const Pose2 pose = slam_.localize(odom, ranges);
    if (cfg_.track_obstacles) {
        // Кандидаты в подвижные объекты — отражения из мест, которые карта знает
        // как свободные; трекер собирает их в объекты и оценивает скорости.
        const Pose2 sensor = slam_.sensor_pose();
        slam_.map().find_transient_hits(sensor, slam_.ray_directions(), ranges, slam_.scan().range_max, transient_);
        tracker_.update(sensor, slam_.scan(), ranges, transient_, cfg_.control_dt, ray_flags_);
    } else {
        ray_flags_.assign(ranges.size(), ray_flag::Static);
    }
    slam_.integrate(ranges, ray_flags_);

    if (!has_goal_) {
        status_ = NavStatus::Idle;
        return {};
    }
    if (distance(pose.position(), goal_) < cfg_.goal_tolerance) {
        status_ = NavStatus::Reached;
        command_ = {};
        return {};
    }

    // Стартовая пауза (если задана). Сразу после включения карты вокруг ещё нет, и
    // трекеру не с чем сравнивать скан: объект, который едет навстречу, неотличим
    // от стены. На время паузы навигатор стоит и только наблюдает.
    if (time_ <= cfg_.startup_time) {
        command_ = {};
        command_.feasible = true;
        progress_time_ = time_;  // ожидание — не застревание
        return {};
    }

    // ---- 2. Глобальный маршрут ----
    // Перепланирование: по таймеру (вдруг открылся путь короче), при отсутствии
    // маршрута и — с ограничением частоты — когда текущий маршрут перекрыт.
    const bool due = time_ >= next_plan_time_ || (path_.empty() && status_ != NavStatus::NoPath);
    const bool can_react = time_ - last_plan_time_ >= cfg_.replan_min_period;
    const bool no_path_retry = status_ == NavStatus::NoPath && time_ - last_plan_time_ >= 2.0 * cfg_.replan_min_period;
    if (due || no_path_retry || (can_react && path_blocked())) replan();
    carrot_ = path_.empty() ? goal_ : lookahead_point(pose.position());

    // ---- 3. Локальная картина в СК робота ----
    const double c = std::cos(pose.theta), s = std::sin(pose.theta);
    const ScanGeometry& scan = slam_.scan();
    const double reach = std::min(scan.range_max - 1e-3, cfg_.obstacle_range);
    // В локальную картину идут все точки скана, включая точки подвижных объектов:
    // пока трек не подтверждён или прогноз неточен, объект остаётся препятствием
    // хотя бы в своём текущем положении. Прогноз движения добавляется поверх.
    obstacle_points_.clear();
    const std::size_t n = std::min(ranges.size(), ray_dirs_robot_.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (ranges[i] >= reach) continue;
        obstacle_points_.push_back(Vec2{scan.mount.x, scan.mount.y} + ray_dirs_robot_[i] * static_cast<double>(ranges[i]));
    }
    moving_.clear();
    for (const Track& track : tracker_.tracks()) {
        if (!track.moving) continue;
        moving_.push_back({rotate_inv(track.position - pose.position(), c, s), rotate_inv(track.velocity, c, s), track.radius});
    }
    const Vec2 target = rotate_inv(carrot_ - pose.position(), c, s);

    // ---- 4. Контроль продвижения и восстановление ----
    if (distance(pose.position(), progress_anchor_) > 0.15) {
        progress_anchor_ = pose.position();
        progress_time_ = time_;
    } else if (time_ - progress_time_ > cfg_.stuck_time && time_ >= backup_until_) {
        backup_until_ = time_ + cfg_.backup_time;  // застряли: отъезжаем и строим маршрут заново
        progress_time_ = time_ + cfg_.backup_time;
        next_plan_time_ = time_ + cfg_.backup_time;
    }
    if (time_ < backup_until_) {
        status_ = NavStatus::Recovering;
        // Назад можно, только если сзади свободно; иначе разворот на месте.
        double rear = std::numeric_limits<double>::infinity();
        const double lane = cfg_.dwa.robot_radius + 0.05;
        for (const Vec2& p : obstacle_points_) {
            if (p.x < 0.0 && std::abs(p.y) < lane) rear = std::min(rear, -p.x);
        }
        command_ = {};
        if (rear > cfg_.dwa.robot_radius + 0.15) {
            command_.v = -cfg_.backup_speed;
        } else {
            command_.w = std::copysign(1.0, target.y);
        }
        return wheels(command_.v, command_.w);
    }
    if (status_ == NavStatus::Recovering) status_ = path_.empty() ? NavStatus::NoPath : NavStatus::Driving;

    // ---- 5. Локальный планировщик ----
    if (status_ == NavStatus::NoPath) {
        // На текущей карте маршрута нет — обычно потому, что проход перекрыт
        // подвижным препятствием. Ехать к цели напролом бессмысленно: стоим и
        // ждём, пока карта обновится (перепланирование идёт каждые полсекунды).
        command_ = {};
        command_.feasible = true;
        progress_time_ = time_;  // ожидание — не застревание
        return {};
    }
    const double bearing = std::atan2(target.y, target.x);
    // Угроза: подвижное препятствие рядом и приближается. Тогда разворот на месте
    // откладывается, а DWA получает право отъехать назад.
    const bool threatened = std::any_of(moving_.begin(), moving_.end(), [&](const MovingObstacle& m) {
        const double dist = norm(m.position);
        const double closing = dist > 1e-9 ? -dot(m.velocity, m.position) / dist : 0.0;
        return dist < m.radius + cfg_.dwa.robot_radius + 1.2 && closing > 0.05;
    });
    if (std::abs(bearing) > cfg_.rotate_threshold && !threatened) {
        // Локальная цель сбоку или сзади. Круглый робот поворачивается на месте,
        // ничего не задевая, поэтому сначала разворот, потом движение: так он не
        // описывает широкую дугу в тесном проходе.
        command_ = {};
        command_.feasible = true;
        command_.w = std::copysign(std::min(cfg_.dwa.max_yaw_rate, 1.5 * std::abs(bearing)), bearing);
        return wheels(0.0, command_.w);
    }
    command_ = dwa_.compute(obstacle_points_, moving_, target, odom_v, odom_w, threatened);
    if (!command_.feasible) {
        // Допустимых траекторий нет (препятствие вплотную): безопасный манёвр —
        // стоя на месте повернуться к локальной цели.
        command_.v = 0.0;
        command_.w = std::abs(bearing) > 0.1 ? std::copysign(0.8, bearing) : 0.0;
    }
    return wheels(command_.v, command_.w);
}

}  // namespace roversim::nav
