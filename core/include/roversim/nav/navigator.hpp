// Навигатор: полный контур автономного движения к цели в неизвестной среде.
//
//     лидар, одометрия
//            │
//            ▼
//   ┌──────────────────┐   поза, карта   ┌──────────────────────┐
//   │ GridSlam         │ ───────────────►│ GridPlanner (Theta*) │
//   │ локализация      │                 │ глобальный маршрут   │
//   │ и карта          │                 └──────────┬───────────┘
//   └────────┬─────────┘                            │ точка маршрута впереди
//            │ скан, поза                           ▼
//   ┌────────▼─────────┐  подвижные      ┌──────────────────────┐
//   │ ObstacleTracker  │ ───────────────►│ DwaPlanner           │──► (v, w) ──► колёса
//   │ скорости объектов│  объекты        │ локальный объезд     │
//   └──────────────────┘                 └──────────────────────┘
//
// Навигатору нужны только показания сенсоров: дальности лидара и одометрия.
// Истинная поза и карта мира не используются, поэтому он одинаково работает и
// внутри процесса симулятора, и через разделяемую память.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "roversim/math.hpp"
#include "roversim/nav/dwa.hpp"
#include "roversim/nav/planner.hpp"
#include "roversim/nav/slam.hpp"
#include "roversim/nav/tracker.hpp"
#include "roversim/robot.hpp"
#include "roversim/sensors.hpp"

namespace roversim::nav {

struct NavigatorConfig {
    SlamConfig slam;
    PlannerConfig planner;
    DwaConfig dwa;
    TrackerConfig tracker;

    double control_dt = 0.05;      ///< период вызова update() [с]
    double wheel_radius = 0.05;    ///< кинематика привода — для перевода (v, w) в уставки колёс
    double track_width = 0.30;
    double max_wheel_speed = 20.0;

    double replan_period = 2.0;    ///< маршрут перестраивается не реже этого интервала [с]...
    double replan_min_period = 0.25; ///< ...и не чаще этого, когда текущий маршрут оказался перекрыт [с]
    double path_check_distance = 3.0; ///< на каком участке маршрута впереди проверяется проходимость [м]
    double lookahead = 0.7;        ///< расстояние до локальной цели вдоль маршрута [м]
    double goal_tolerance = 0.20;  ///< цель достигнута, если оценка позы ближе [м]
    double obstacle_range = 3.0;   ///< лучи длиннее в локальном планировании не участвуют [м]
    double rotate_threshold = 1.0; ///< локальная цель под большим углом [рад] — сначала разворот на месте
    bool track_obstacles = true;   ///< сопровождать подвижные объекты и прогнозировать их движение
    double startup_time = 0.0;     ///< после включения навигатор столько секунд стоит и строит карту
                                   ///< окружения (без неё не отличить стену от едущего навстречу
                                   ///< объекта); 0 — движение начинается сразу
    double stuck_time = 2.5;       ///< нет продвижения столько секунд — манёвр восстановления
    double backup_time = 1.0;      ///< длительность отъезда назад [с]
    double backup_speed = 0.12;    ///< скорость отъезда назад [м/с]
};

/// Геометрия скана по модели лидара симулятора.
ScanGeometry scan_geometry(const Lidar2D& lidar);
/// Настройки навигатора, согласованные с параметрами робота.
NavigatorConfig navigator_config(const RobotParams& robot, double control_dt);

enum class NavStatus : int {
    Idle = 0,        ///< цель не задана
    Driving = 1,     ///< едет по маршруту
    Reached = 2,     ///< цель достигнута
    NoPath = 3,      ///< маршрута на текущей карте нет (едет к цели напрямую и ждёт новых данных)
    Recovering = 4,  ///< манёвр восстановления после застревания
};

class Navigator {
 public:
    /// initial_pose задаёт СК карты; обычно это начальная поза одометрии.
    Navigator(const NavigatorConfig& config, const ScanGeometry& scan, const Pose2& initial_pose);

    /// Цель в СК карты.
    void set_goal(Vec2 goal);

    /// Один такт управления.
    ///   ranges   — скан лидара;
    ///   odom     — поза по одометрии; odom_v, odom_w — скорости по одометрии.
    /// Возвращает уставки угловых скоростей колёс [рад/с].
    WheelCommand update(std::span<const float> ranges, const Pose2& odom, double odom_v, double odom_w);

    NavStatus status() const noexcept { return status_; }
    const Pose2& pose() const noexcept { return slam_.pose(); }  ///< оценка позы в СК карты
    Vec2 goal() const noexcept { return goal_; }
    Vec2 carrot() const noexcept { return carrot_; }             ///< текущая локальная цель
    std::span<const Vec2> path() const noexcept { return path_; }
    std::span<const Track> tracks() const noexcept { return tracker_.tracks(); }
    const GridSlam& slam() const noexcept { return slam_; }
    const OccupancyGrid& map() const noexcept { return slam_.map(); }
    const GridPlanner& planner() const noexcept { return planner_; }
    const DwaPlanner& dwa() const noexcept { return dwa_; }
    const DwaResult& last_command() const noexcept { return command_; }
    int replans() const noexcept { return replans_; }
    const NavigatorConfig& config() const noexcept { return cfg_; }

 private:
    void replan();
    bool path_blocked() const;
    Vec2 lookahead_point(Vec2 position);
    WheelCommand wheels(double v, double w) const noexcept;

    NavigatorConfig cfg_;
    GridSlam slam_;
    GridPlanner planner_;
    DwaPlanner dwa_;
    ObstacleTracker tracker_;

    Vec2 goal_{};
    bool has_goal_ = false;
    NavStatus status_ = NavStatus::Idle;
    std::vector<Vec2> path_;
    std::size_t path_cursor_ = 0;  ///< отрезок маршрута, ближайший к роботу
    Vec2 carrot_{};
    DwaResult command_{};
    int replans_ = 0;

    double time_ = 0.0;
    double next_plan_time_ = 0.0;
    double last_plan_time_ = -1e9;
    double backup_until_ = -1.0;
    Vec2 progress_anchor_{};     ///< от этой точки отсчитывается продвижение
    double progress_time_ = 0.0;

    std::vector<std::uint8_t> transient_;     ///< лучи-кандидаты в подвижные объекты
    std::vector<std::uint8_t> ray_flags_;     ///< отметки лучей для карты (ray_flag)
    std::vector<Vec2> obstacle_points_;      ///< точки скана в СК робота
    std::vector<MovingObstacle> moving_;     ///< подвижные объекты в СК робота
    std::vector<Vec2> ray_dirs_robot_;       ///< направления лучей в СК робота (с учётом установки сенсора)
};

}  // namespace roversim::nav
