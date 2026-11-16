#include "roversim/simulator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace roversim {
namespace {

// Потоки ГСЧ внутри одного seed эпизода: у каждого источника случайности свой.
enum : std::uint64_t { kStreamEpisode = 0, kStreamLidar = 1, kStreamImu = 2, kStreamOdom = 3, kStreamObstacles = 4 };

// Импульс стен за шаг управления [Н·с], начиная с которого шаг считается столкновением.
constexpr double kTouchImpulse = 1e-9;

const SimConfig& validated(const SimConfig& cfg) {
    if (!(cfg.physics_dt > 0.0)) throw std::invalid_argument("SimConfig: physics_dt must be > 0");
    if (cfg.substeps < 1) throw std::invalid_argument("SimConfig: substeps must be >= 1");
    if (!(cfg.task.goal_radius > 0.0)) throw std::invalid_argument("TaskConfig: goal_radius must be > 0");
    if (cfg.task.spawn_clearance < 0.0) throw std::invalid_argument("TaskConfig: spawn_clearance must be >= 0");
    if (!(cfg.task.lookahead > 0.0)) throw std::invalid_argument("TaskConfig: lookahead must be > 0");
    const ObstacleConfig& o = cfg.obstacles;
    if (o.count < 0 || !(o.radius > 0.0) || o.speed_min < 0.0 || o.speed_max < o.speed_min) {
        throw std::invalid_argument("ObstacleConfig: need count >= 0, radius > 0, 0 <= speed_min <= speed_max");
    }
    return cfg;
}

}  // namespace

Simulator::Simulator(const SimConfig& config, std::shared_ptr<const World> world, Buffers buffers,
                     std::shared_ptr<const NavGrid> nav)
    : cfg_(validated(config)),
      world_(std::move(world)),
      nav_(std::move(nav)),
      drive_(cfg_.robot),
      lidar_model_(cfg_.lidar),
      imu_(cfg_.imu, cfg_.control_dt()),
      odometry_(cfg_.odom, cfg_.robot) {
    if (!world_) throw std::invalid_argument("Simulator: world is null");
    if (!world_->finalized()) throw std::invalid_argument("Simulator: call World::finalize() first");

    num_rays_ = static_cast<std::size_t>(cfg_.lidar.num_rays);
    if (buffers.lidar != nullptr) {
        lidar_ = buffers.lidar;
    } else {
        lidar_store_.assign(num_rays_, 0.0F);
        lidar_ = lidar_store_.data();
    }
    if (buffers.proprio != nullptr) {
        proprio_ = buffers.proprio;
    } else {
        proprio_store_.assign(obs::Dim, 0.0F);
        proprio_ = proprio_store_.data();
    }
    if (buffers.state != nullptr) {
        state_buf_ = buffers.state;
    } else {
        state_store_.assign(gt::Dim, 0.0);
        state_buf_ = state_store_.data();
    }

    // Упреждающие контакты нужны только с тем, до чего робот может доехать за
    // один шаг физики; удвоенный запас покрывает занос и буксование.
    contact_margin_ = 2.0 * drive_.max_linear_speed() * cfg_.physics_dt + 1e-3;
    obstacle_margin_ = contact_margin_ + cfg_.obstacles.speed_max * cfg_.physics_dt;

    if (nav_) nav_revision_ = world_->revision();
    reset(0);
}

void Simulator::refresh_nav() {
    if (!world_->finalized()) {
        throw std::logic_error("Simulator: world was modified; call World::finalize() before reset()");
    }
    if (!nav_ || nav_revision_ != world_->revision()) {
        nav_ = std::make_shared<NavGrid>(*world_, nav_radius());
        nav_revision_ = world_->revision();
    }
    if (nav_->free_count() == 0) {
        throw std::invalid_argument("Simulator: the world has no free space large enough for the robot");
    }
}

void Simulator::set_world(std::shared_ptr<const World> world, std::shared_ptr<const NavGrid> nav) {
    if (!world) throw std::invalid_argument("Simulator::set_world: world is null");
    if (!world->finalized()) throw std::invalid_argument("Simulator::set_world: call World::finalize() first");
    world_ = std::move(world);
    nav_ = std::move(nav);
    nav_revision_ = world_->revision();
    field_valid_ = false;
    needs_reset_ = true;
}

Vec2 Simulator::sample_point(Rng& rng) const {
    const Vec2 center = nav_->sample(rng);
    // Случайный сдвиг внутри клетки, чтобы позиции не ложились на решётку.
    const double cell = world_->cell_size();
    const Vec2 shifted = center + Vec2{rng.uniform(-0.5, 0.5) * cell, rng.uniform(-0.5, 0.5) * cell};
    return world_->is_free(shifted, cfg_.robot.radius + 0.5 * cfg_.task.spawn_clearance) ? shifted : center;
}

void Simulator::reset(std::uint64_t seed) {
    refresh_nav();
    Rng rng = Rng::stream(seed, kStreamEpisode);
    const Vec2 position = sample_point(rng);
    const Pose2 start{position.x, position.y, rng.uniform(-kPi, kPi)};

    // Цель — первая точка не ближе min_goal_distance. В тесном мире такой может
    // не найтись; тогда берётся самая дальняя из опробованных.
    Vec2 goal = position;
    double best = -1.0;
    for (int attempt = 0; attempt < 64; ++attempt) {
        const Vec2 candidate = sample_point(rng);
        const double d = distance(candidate, position);
        if (d > best) {
            best = d;
            goal = candidate;
        }
        if (d >= cfg_.task.min_goal_distance) break;
    }
    begin_episode(seed, start, goal);
}

void Simulator::reset(std::uint64_t seed, const Pose2& start, Vec2 goal) {
    refresh_nav();
    if (!std::isfinite(start.x) || !std::isfinite(start.y) || !std::isfinite(start.theta) ||
        !std::isfinite(goal.x) || !std::isfinite(goal.y)) {
        throw std::invalid_argument("Simulator::reset: start and goal must be finite");
    }
    if (!world_->is_free(start.position(), cfg_.robot.radius)) {
        throw std::invalid_argument("Simulator::reset: start pose collides with an obstacle");
    }
    begin_episode(seed, start, goal);
}

void Simulator::begin_episode(std::uint64_t seed, const Pose2& start, Vec2 goal) {
    robot_ = {};
    robot_.pose = {start.x, start.y, wrap_angle(start.theta)};
    start_ = start.position();
    goal_ = goal;
    steps_ = 0;
    ++episode_;
    result_ = {};
    contact_impulse_ = 0.0;
    needs_reset_ = false;

    rng_lidar_ = Rng::stream(seed, kStreamLidar);
    rng_imu_ = Rng::stream(seed, kStreamImu);
    rng_odom_ = Rng::stream(seed, kStreamOdom);
    rng_obstacles_ = Rng::stream(seed, kStreamObstacles);
    imu_.reset(robot_, rng_imu_);
    odometry_.reset(robot_, rng_odom_);
    spawn_obstacles();

    field_valid_ = false;
    if (cfg_.task.geodesic || cfg_.task.path_hint) ensure_field();
    metric_ = distance(start_, goal_);
    metric_ = goal_metric();
    sense();
}

void Simulator::ensure_field() const {
    if (!field_valid_) {
        nav_->distance_field(goal_, field_, next_cell_);
        field_valid_ = true;
    }
}

double Simulator::goal_metric() const {
    const Vec2 position = robot_.pose.position();
    if (!cfg_.task.geodesic) return distance(position, goal_);
    // Если поле в этой точке не определено (робот вжат в угол), считаем, что
    // прогресса нет: резкий скачок метрики дал бы ложный всплеск награды.
    const double d = nav_->lookup(field_, position);
    return std::isfinite(d) ? d : metric_;
}

const StepResult& Simulator::step(WheelCommand command) {
    if (needs_reset_) throw std::logic_error("Simulator::step: call reset() after set_world()");
    if (!std::isfinite(command.left)) command.left = 0.0;
    if (!std::isfinite(command.right)) command.right = 0.0;

    const double dt = cfg_.physics_dt;
    const double radius = cfg_.robot.radius;
    std::array<Contact, DiffDrive::kMaxContacts> contacts;
    double impulse = 0.0;
    bool touched = false;  // подвижное препятствие коснулось робота
    for (int k = 0; k < cfg_.substeps; ++k) {
        if (!obstacles_.empty()) touched = move_obstacles(dt) || touched;
        const Vec2 position = robot_.pose.position();
        std::size_t count = world_->collide_circle(position, radius, contact_margin_, contacts);
        // Подвижное препятствие для робота — такой же контакт, как стена. Схема
        // расщеплённая: на этом шаге препятствия уже сдвинулись (move_obstacles), и
        // их движение целиком отражено в зазоре, поэтому поправка на скорость
        // препятствия не нужна. Робот может догонять уезжающее препятствие, но не
        // обязан отступать перед надвигающимся: оно его не толкает, а отскакивает.
        for (const DynamicObstacle& obstacle : obstacles_) {
            if (count == contacts.size()) break;
            const Vec2 delta = position - obstacle.position;
            const double dist = norm(delta);
            const double gap = dist - radius - obstacle.radius;
            if (gap >= obstacle_margin_) continue;
            const Vec2 normal = dist > 1e-9 ? delta * (1.0 / dist) : Vec2{1.0, 0.0};
            contacts[count++] = {normal, gap};
        }
        SubstepInfo info;
        drive_.step(robot_, command, std::span<const Contact>(contacts.data(), count), dt, &info);
        if (count > 0) resolve_penetration();
        imu_.accumulate(robot_, dt);
        impulse += info.contact_impulse;
    }
    ++steps_;
    contact_impulse_ = impulse;

    const double previous_metric = metric_;
    metric_ = goal_metric();
    sense();

    const TaskConfig& task = cfg_.task;
    StepResult r;
    r.collision = impulse > kTouchImpulse || touched;
    r.success = distance(robot_.pose.position(), goal_) < task.goal_radius;
    r.reward = task.progress_weight * (previous_metric - metric_) - task.step_penalty;
    if (r.collision) r.reward -= task.collision_penalty;
    if (r.success) r.reward += task.success_reward;
    r.terminated = r.success || (r.collision && task.terminate_on_collision);
    r.truncated = !r.terminated && task.max_steps > 0 && steps_ >= static_cast<std::uint64_t>(task.max_steps);
    result_ = r;
    return result_;
}

// Позиционная коррекция после шага физики. Решатель запрещает сближение со
// стеной вдоль нормали, но это линейное условие, а поза интегрируется по дуге:
// при езде с поворотом у самой стены корпус заходит в неё на величину порядка
// v·w·dt²/2 (десятки микрон). Проекция возвращает его на поверхность.
void Simulator::resolve_penetration() {
    std::array<Contact, DiffDrive::kMaxContacts> contacts;
    Vec2 position = robot_.pose.position();
    for (int pass = 0; pass < 4; ++pass) {
        const std::size_t count = world_->collide_circle(position, cfg_.robot.radius, 0.0, contacts);
        if (count == 0) break;
        const Vec2 pass_start = position;
        for (std::size_t i = 0; i < count; ++i) {
            // Зазор с учётом сдвигов, уже сделанных в этом проходе для других контактов.
            const double gap = contacts[i].gap + dot(contacts[i].normal, position - pass_start);
            if (gap < 0.0) position += contacts[i].normal * (-gap);
        }
    }
    robot_.pose.x = position.x;
    robot_.pose.y = position.y;
}

void Simulator::spawn_obstacles() {
    obstacles_.clear();
    const ObstacleConfig& cfg = cfg_.obstacles;
    for (int i = 0; i < cfg.count; ++i) {
        // Место выбирается с ограничениями: свободно, не у старта (робот должен успеть
        // среагировать), не на цели и не внутри другого препятствия. В тесном мире
        // подходящего места может не найтись — тогда препятствий будет меньше.
        for (int attempt = 0; attempt < 40; ++attempt) {
            const Vec2 p = nav_->sample(rng_obstacles_);
            if (!world_->is_free(p, cfg.radius + 0.05)) continue;
            if (distance(p, start_) < 1.0 + cfg.radius || distance(p, goal_) < 0.5 + cfg.radius) continue;
            const bool overlaps = std::any_of(obstacles_.begin(), obstacles_.end(), [&](const DynamicObstacle& other) {
                return distance(p, other.position) < cfg.radius + other.radius + 0.1;
            });
            if (overlaps) continue;
            const double speed = rng_obstacles_.uniform(cfg.speed_min, cfg.speed_max);
            const double heading = rng_obstacles_.uniform(-kPi, kPi);
            obstacles_.push_back({p, {speed * std::cos(heading), speed * std::sin(heading)}, cfg.radius});
            break;
        }
    }
}

// Шаг подвижных препятствий. Каждое едет по прямой и отражается от того, во что
// упирается. Возвращает true, если какое-то из них коснулось робота.
bool Simulator::move_obstacles(double dt) {
    bool touched = false;
    const Vec2 robot = robot_.pose.position();
    const double robot_radius = cfg_.robot.radius;
    const auto reflect = [](DynamicObstacle& obstacle, Vec2 normal) {
        const double approach = dot(obstacle.velocity, normal);
        if (approach < 0.0) obstacle.velocity = obstacle.velocity - normal * (2.0 * approach);
    };

    Contact contacts[4];
    for (DynamicObstacle& obstacle : obstacles_) {
        Vec2 next = obstacle.position + obstacle.velocity * dt;

        // Стены: зеркальное отражение от суммарной нормали плюс небольшой случайный
        // поворот, чтобы траектории не замыкались в короткие циклы.
        const std::size_t count = world_->collide_circle(next, obstacle.radius, 0.0, contacts);
        if (count > 0) {
            Vec2 normal{};
            for (std::size_t i = 0; i < count; ++i) normal += contacts[i].normal;
            const double length = norm(normal);
            if (length > 1e-9) {
                normal = normal * (1.0 / length);
                reflect(obstacle, normal);
                const double turn = rng_obstacles_.uniform(-0.3, 0.3);
                obstacle.velocity = rotate(obstacle.velocity, std::cos(turn), std::sin(turn));
                reflect(obstacle, normal);  // после поворота скорость не должна вести обратно в стену
            }
            next = obstacle.position;  // этот шаг препятствие пропускает: прежняя позиция свободна
        }

        // Робот. Препятствие, которое в него упёрлось, отскакивает и остаётся на
        // месте: оно не въезжает в робота и не толкает его (иначе могло бы вдавить
        // в стену). Препятствию, которое уже удаляется, двигаться не мешаем.
        const double reach = obstacle.radius + robot_radius;
        Vec2 delta = next - robot;
        double dist = norm(delta);
        if (dist < reach) {
            const Vec2 normal = dist > 1e-9 ? delta * (1.0 / dist) : Vec2{1.0, 0.0};
            if (dot(obstacle.velocity, normal) < 0.0) {
                reflect(obstacle, normal);
                next = obstacle.position;
                touched = true;
            }
            // Остаточное перекрытие (погрешности интегрирования) убирается сдвигом
            // препятствия — если по ту сторону свободно.
            delta = next - robot;
            dist = norm(delta);
            if (dist < reach && dist > 1e-9) {
                const Vec2 separated = robot + delta * (reach / dist);
                if (world_->is_free(separated, obstacle.radius)) next = separated;
            }
        }
        obstacle.position = next;
    }

    // Друг от друга препятствия отскакивают: каждое отражается от линии центров и
    // сохраняет свою скорость по модулю (иначе после нескольких столкновений одни
    // почти остановились бы, а другие разогнались за пределы заданного диапазона).
    for (std::size_t i = 0; i < obstacles_.size(); ++i) {
        for (std::size_t j = i + 1; j < obstacles_.size(); ++j) {
            DynamicObstacle& a = obstacles_[i];
            DynamicObstacle& b = obstacles_[j];
            const Vec2 delta = b.position - a.position;
            const double dist = norm(delta);
            if (dist >= a.radius + b.radius || dist < 1e-9) continue;
            const Vec2 normal = delta * (1.0 / dist);  // от a к b
            if (dot(b.velocity - a.velocity, normal) < 0.0) {
                reflect(a, normal * -1.0);
                reflect(b, normal);
            }
        }
    }
    return touched;
}

void Simulator::sense() {
    obstacle_circles_.clear();
    for (const DynamicObstacle& obstacle : obstacles_) obstacle_circles_.push_back({obstacle.position, obstacle.radius});
    lidar_model_.scan(*world_, robot_.pose, obstacle_circles_, rng_lidar_, {lidar_, num_rays_});
    const ImuSample imu = imu_.sample(robot_, rng_imu_);
    // На нулевом шаге эпизода колёса ещё не вращались: берём начальное состояние одометрии.
    const OdomSample& odom = steps_ == 0 ? odometry_.last() : odometry_.update(robot_, cfg_.control_dt(), rng_odom_);

    // Цель в связанной СК робота. Поза здесь истинная: предполагается, что
    // задачу локализации решает отдельная система (постановка «GPS + компас»).
    const double c = std::cos(robot_.pose.theta), s = std::sin(robot_.pose.theta);
    const Vec2 to_goal = rotate_inv(goal_ - robot_.pose.position(), c, s);
    const double goal_distance = norm(to_goal);
    const bool at_goal = goal_distance < 1e-9;

    // Подсказка маршрута: направление на точку кратчайшего пути в lookahead метрах
    // впереди. Так глобальный планировщик сообщает локальному, куда ехать; в
    // лабиринте это направление, в отличие от направления на цель, не упирается в стену.
    Vec2 hint = at_goal ? Vec2{1.0, 0.0} : to_goal * (1.0 / goal_distance);
    if (cfg_.task.path_hint) {
        const Vec2 carrot = nav_->lookahead(next_cell_, robot_.pose.position(), goal_, cfg_.task.lookahead);
        const Vec2 to_carrot = rotate_inv(carrot - robot_.pose.position(), c, s);
        const double carrot_distance = norm(to_carrot);
        if (carrot_distance > 1e-9) hint = to_carrot * (1.0 / carrot_distance);
    }

    float* p = proprio_;
    p[obs::GyroZ] = static_cast<float>(imu.gyro_z);
    p[obs::AccelX] = static_cast<float>(imu.accel_x);
    p[obs::AccelY] = static_cast<float>(imu.accel_y);
    p[obs::WheelVelLeft] = static_cast<float>(odom.wheel_velocity[0]);
    p[obs::WheelVelRight] = static_cast<float>(odom.wheel_velocity[1]);
    p[obs::OdomV] = static_cast<float>(odom.v);
    p[obs::OdomW] = static_cast<float>(odom.w);
    p[obs::OdomX] = static_cast<float>(odom.pose.x);
    p[obs::OdomY] = static_cast<float>(odom.pose.y);
    p[obs::OdomTheta] = static_cast<float>(odom.pose.theta);
    p[obs::GoalDistance] = static_cast<float>(goal_distance);
    p[obs::GoalCos] = static_cast<float>(at_goal ? 1.0 : to_goal.x / goal_distance);
    p[obs::GoalSin] = static_cast<float>(at_goal ? 0.0 : to_goal.y / goal_distance);
    p[obs::PathCos] = static_cast<float>(hint.x);
    p[obs::PathSin] = static_cast<float>(hint.y);

    double* g = state_buf_;
    g[gt::X] = robot_.pose.x;
    g[gt::Y] = robot_.pose.y;
    g[gt::Theta] = robot_.pose.theta;
    g[gt::Vx] = robot_.velocity.x;
    g[gt::Vy] = robot_.velocity.y;
    g[gt::YawRate] = robot_.yaw_rate;
    g[gt::WheelSpeedLeft] = robot_.wheel_speed[0];
    g[gt::WheelSpeedRight] = robot_.wheel_speed[1];
    g[gt::GoalX] = goal_.x;
    g[gt::GoalY] = goal_.y;
    g[gt::ContactForce] = contact_impulse_ / cfg_.control_dt();
    g[gt::PathDistance] = metric_;
}

double Simulator::shortest_path_length() const {
    ensure_field();
    return nav_->lookup(field_, start_);
}

std::vector<Vec2> Simulator::shortest_path() const {
    ensure_field();
    return nav_->path(next_cell_, robot_.pose.position(), goal_);
}

}  // namespace roversim
