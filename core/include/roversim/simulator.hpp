// Симулятор: физика + сенсоры + задача навигации за интерфейсом в стиле Gym.
//
//     Simulator sim(config, world);
//     sim.reset(seed);
//     while (...) {
//         StepResult r = sim.step({left, right});   // уставки скоростей колёс [рад/с]
//         std::span<const float> scan = sim.lidar();
//         if (r.terminated || r.truncated) sim.reset(next_seed);
//     }
//
// Наблюдения лежат в непрерывных буферах, которые симулятор пишет напрямую:
// собственных либо внешних (разделяемая память, пакетный массив VecSimulator).
// Чтение наблюдения — это чтение указателя, без копирований и выделений памяти.
//
// Детерминизм: при одинаковых конфигурации, мире, seed и последовательности
// действий траектория и показания сенсоров совпадают бит в бит. Симулятор не
// обращается к часам и не использует потоки.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "roversim/navgrid.hpp"
#include "roversim/robot.hpp"
#include "roversim/sensors.hpp"
#include "roversim/world.hpp"

namespace roversim {

/// Индексы вектора наблюдения `proprio` (float32): всё, что робот знает о себе.
namespace obs {
enum : int {
    GyroZ = 0,      ///< IMU: угловая скорость [рад/с]
    AccelX,         ///< IMU: кажущееся ускорение вперёд [м/с²]
    AccelY,         ///< IMU: кажущееся ускорение влево [м/с²]
    WheelVelLeft,   ///< энкодер: скорость левого колеса [рад/с]
    WheelVelRight,  ///< энкодер: скорость правого колеса [рад/с]
    OdomV,          ///< одометрия: линейная скорость [м/с]
    OdomW,          ///< одометрия: угловая скорость [рад/с]
    OdomX,          ///< одометрия: счисленная поза
    OdomY,
    OdomTheta,
    GoalDistance,   ///< расстояние до цели по прямой [м]
    GoalCos,        ///< косинус и синус пеленга цели в связанной СК
    GoalSin,
    PathCos,        ///< косинус и синус пеленга ближайшей точки маршрута:
    PathSin,        ///< направление, в котором кратчайший путь ведёт к цели
    Dim
};
}  // namespace obs

/// Индексы вектора истинного состояния `state` (float64): для наград, метрик и
/// отладки. Политике, которую планируется переносить на реального робота, эти
/// данные недоступны.
namespace gt {
enum : int {
    X = 0,
    Y,
    Theta,
    Vx,               ///< скорость в связанной СК: вперёд [м/с]
    Vy,               ///< скорость в связанной СК: влево [м/с]
    YawRate,
    WheelSpeedLeft,   ///< истинные угловые скорости колёс [рад/с]
    WheelSpeedRight,
    GoalX,
    GoalY,
    ContactForce,     ///< средняя за шаг сила давления стен [Н]
    PathDistance,     ///< оставшийся путь до цели в метрике награды [м]
    Dim
};
}  // namespace gt

/// Задача «доехать до точки» и её награда:
///   r = progress_weight·(d_prev − d) − step_penalty
///       − collision_penalty·[касание стены] + success_reward·[цель достигнута].
struct TaskConfig {
    double goal_radius = 0.30;           ///< цель достигнута, если центр робота ближе [м]
    int max_steps = 600;                 ///< лимит шагов эпизода (truncation); 0 — без лимита
    bool terminate_on_collision = true;  ///< завершать эпизод при касании стены
    double min_goal_distance = 2.0;      ///< минимальная дистанция от старта до случайной цели [м]
    double spawn_clearance = 0.10;       ///< запас до стен при выборе старта и цели [м]
    bool geodesic = true;                ///< прогресс по кратчайшему пути (иначе по прямой)
    bool path_hint = true;               ///< давать в наблюдении направление по маршруту (PathCos/PathSin);
                                         ///< false — туда пишется направление на цель по прямой
    double lookahead = 0.8;              ///< на каком расстоянии по маршруту берётся точка подсказки [м]
    double progress_weight = 1.0;
    double step_penalty = 0.01;
    double collision_penalty = 5.0;
    double success_reward = 10.0;
};

/// Подвижные препятствия: круги, которые ездят по прямой с постоянной скоростью
/// и отражаются от стен, друг от друга и от робота. В статической карте их нет:
/// робот видит их только лидаром.
struct ObstacleConfig {
    int count = 0;            ///< сколько препятствий создавать в каждом эпизоде
    double radius = 0.20;     ///< радиус [м]
    double speed_min = 0.20;  ///< диапазон скоростей [м/с]
    double speed_max = 0.50;
};

struct DynamicObstacle {
    Vec2 position;
    Vec2 velocity;
    double radius = 0.0;
};

struct SimConfig {
    double physics_dt = 0.005;  ///< шаг физики [с]
    int substeps = 10;          ///< шагов физики на один step()
    RobotParams robot;
    LidarConfig lidar;
    ImuConfig imu;
    OdomConfig odom;
    TaskConfig task;
    ObstacleConfig obstacles;

    /// Период управления и опроса сенсоров [с].
    double control_dt() const noexcept { return physics_dt * substeps; }
};

struct StepResult {
    double reward = 0.0;
    bool terminated = false;  ///< эпизод завершён: цель или столкновение
    bool truncated = false;   ///< эпизод прерван по лимиту шагов
    bool collision = false;   ///< на этом шаге робот касался препятствия
    bool success = false;     ///< цель достигнута
};

class Simulator {
 public:
    /// Внешняя память под наблюдения. Нулевой указатель — симулятор выделит свою.
    /// Размеры: lidar — num_rays, proprio — obs::Dim, state — gt::Dim.
    struct Buffers {
        float* lidar = nullptr;
        float* proprio = nullptr;
        double* state = nullptr;
    };

    /// `nav` можно передать, чтобы несколько симуляторов разделяли одну
    /// навигационную сетку; иначе она строится здесь.
    /// Конструктор выполняет reset(0), так что наблюдения сразу корректны.
    Simulator(const SimConfig& config, std::shared_ptr<const World> world, Buffers buffers,
              std::shared_ptr<const NavGrid> nav = nullptr);
    Simulator(const SimConfig& config, std::shared_ptr<const World> world)
        : Simulator(config, std::move(world), Buffers{}) {}

    Simulator(const Simulator&) = delete;
    Simulator& operator=(const Simulator&) = delete;
    Simulator(Simulator&&) noexcept = default;
    Simulator& operator=(Simulator&&) noexcept = default;

    /// Новый эпизод: случайные старт и цель (определяются только seed и миром).
    void reset(std::uint64_t seed);
    /// Новый эпизод с заданными стартом и целью; seed задаёт шум сенсоров.
    /// std::invalid_argument, если стартовая поза пересекает препятствие.
    void reset(std::uint64_t seed, const Pose2& start, Vec2 goal);

    /// Один шаг управления: substeps шагов физики, затем опрос сенсоров.
    const StepResult& step(WheelCommand command);

    /// Заменяет мир (например, новый лабиринт на каждый эпизод). После замены
    /// обязателен reset(): до него step() бросает std::logic_error.
    void set_world(std::shared_ptr<const World> world, std::shared_ptr<const NavGrid> nav = nullptr);

    // ---- Наблюдения (обновляются в reset() и step()) ----
    std::span<const float> lidar() const noexcept { return {lidar_, num_rays_}; }
    std::span<const float> proprio() const noexcept { return {proprio_, obs::Dim}; }
    std::span<const double> state() const noexcept { return {state_buf_, gt::Dim}; }
    const StepResult& last_result() const noexcept { return result_; }

    // ---- Состояние эпизода ----
    double time() const noexcept { return static_cast<double>(steps_) * cfg_.control_dt(); }
    std::uint64_t step_count() const noexcept { return steps_; }
    std::uint64_t episode() const noexcept { return episode_; }
    const RobotState& robot_state() const noexcept { return robot_; }
    const Pose2& odom_pose() const noexcept { return odometry_.last().pose; }
    Vec2 goal() const noexcept { return goal_; }
    /// Подвижные препятствия в их текущем состоянии.
    std::span<const DynamicObstacle> obstacles() const noexcept { return obstacles_; }
    /// Длина кратчайшего пути от старта до цели [м] (для метрики SPL); +inf, если пути нет.
    double shortest_path_length() const;
    /// Кратчайший путь от текущего положения до цели.
    std::vector<Vec2> shortest_path() const;

    // ---- Модель ----
    const SimConfig& config() const noexcept { return cfg_; }
    const World& world() const noexcept { return *world_; }
    std::shared_ptr<const World> world_ptr() const noexcept { return world_; }
    const DiffDrive& drive() const noexcept { return drive_; }
    const Lidar2D& lidar_model() const noexcept { return lidar_model_; }

 private:
    void begin_episode(std::uint64_t seed, const Pose2& start, Vec2 goal);
    void refresh_nav();
    void ensure_field() const;
    Vec2 sample_point(Rng& rng) const;
    double goal_metric() const;
    void resolve_penetration();
    void spawn_obstacles();
    bool move_obstacles(double dt);
    void sense();
    double nav_radius() const noexcept { return cfg_.robot.radius + cfg_.task.spawn_clearance; }

    SimConfig cfg_;
    std::shared_ptr<const World> world_;
    std::shared_ptr<const NavGrid> nav_;
    std::uint64_t nav_revision_ = 0;

    DiffDrive drive_;
    Lidar2D lidar_model_;
    Imu imu_;
    WheelOdometry odometry_;

    RobotState robot_{};
    Vec2 start_{};
    Vec2 goal_{};
    StepResult result_{};
    std::uint64_t steps_ = 0;
    std::uint64_t episode_ = 0;
    bool needs_reset_ = false;      ///< мир заменён, эпизод ещё не начат
    double contact_margin_ = 0.0;   ///< дальность упреждающих контактов [м]
    double contact_impulse_ = 0.0;  ///< импульс стен за последний шаг управления [Н·с]
    double metric_ = 0.0;           ///< текущее расстояние до цели в метрике награды [м]

    Rng rng_lidar_, rng_imu_, rng_odom_, rng_obstacles_;

    std::vector<DynamicObstacle> obstacles_;
    std::vector<Circle> obstacle_circles_;  ///< те же препятствия в виде кругов — для лидара
    double obstacle_margin_ = 0.0;          ///< дальность упреждающих контактов с ними [м]

    // Поле расстояний до цели; строится лениво, поэтому mutable.
    mutable std::vector<float> field_;
    mutable std::vector<std::int32_t> next_cell_;  ///< следующая клетка на пути к цели
    mutable bool field_valid_ = false;

    // Буферы наблюдений: указатели смотрят либо в собственные хранилища ниже,
    // либо во внешнюю память. Хранилища — векторы, поэтому при перемещении
    // объекта указатели остаются верными.
    std::size_t num_rays_ = 0;
    float* lidar_ = nullptr;
    float* proprio_ = nullptr;
    double* state_buf_ = nullptr;
    std::vector<float> lidar_store_, proprio_store_;
    std::vector<double> state_store_;
};

}  // namespace roversim
