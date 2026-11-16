// Динамика ровера с дифференциальным приводом.
//
// Модель: плоское твёрдое тело (корпус) + два ведущих колеса на одной оси.
// Состояние шага: скорость корпуса в связанной СК (vx — вперёд, vy — влево),
// угловая скорость рыскания w и угловые скорости колёс. Корпус и колёса связаны
// не кинематически, а через трение шин, поэтому в модели естественно возникают:
//   * инерционность разгона и торможения (масса, момент инерции, приведённый ротор);
//   * ограничение тяги сцеплением и пробуксовка (колесо крутится, корпус стоит);
//   * занос при превышении поперечного сцепления;
//   * упор в стену: корпус остановлен, моторы упираются или буксуют.
//
// Центр масс и центр круга столкновений лежат в середине ведущей оси.
#pragma once

#include <array>
#include <span>

#include "roversim/math.hpp"
#include "roversim/world.hpp"

namespace roversim {

struct RobotParams {
    // --- Геометрия ---
    double radius = 0.20;        ///< радиус корпуса (круг столкновений) [м]
    double wheel_radius = 0.05;  ///< радиус ведущих колёс [м]
    double track_width = 0.30;   ///< колея: расстояние между колёсами [м]

    // --- Инерция ---
    double mass = 10.0;           ///< масса [кг]
    double inertia_z = 0.20;      ///< момент инерции корпуса вокруг вертикали [кг·м²]
    double wheel_inertia = 2e-3;  ///< колесо + приведённый к нему ротор мотора [кг·м²]

    // --- Привод ---
    // Каждое колесо ведёт сервопривод скорости с ограничением момента:
    //   tau = clamp(motor_gain * (w_ref - w), ±motor_max_torque).
    // При motor_gain = tau_stall / w_noload это линейная механическая
    // характеристика коллекторного мотора с ограничением тока.
    double motor_gain = 0.5;        ///< жёсткость привода [Н·м·с/рад]
    double motor_max_torque = 0.6;  ///< предельный момент на колесе [Н·м]
    double max_wheel_speed = 20.0;  ///< ограничение уставки скорости колеса [рад/с]

    // --- Контакт шин с полом ---
    double mu_long = 0.8;               ///< продольное трение: предел тяги и торможения
    double mu_lat = 0.9;                ///< поперечное трение: предел удержания от заноса
    double rolling_resistance = 0.015;  ///< коэффициент сопротивления качению
    double linear_drag = 0.4;           ///< вязкое сопротивление поступательному движению [Н·с/м]
    double angular_drag = 0.02;         ///< вязкое сопротивление вращению [Н·м·с/рад]
    double gravity = 9.81;              ///< [м/с²]
};

struct RobotState {
    Pose2 pose;                           ///< поза в мировой СК
    Vec2 velocity;                        ///< скорость в связанной СК: (вперёд, влево) [м/с]
    double yaw_rate = 0.0;                ///< угловая скорость рыскания [рад/с]
    std::array<double, 2> wheel_speed{};  ///< угловые скорости колёс [левое, правое], рад/с
    std::array<double, 2> wheel_angle{};  ///< накопленные углы поворота колёс [рад] (для энкодеров)
};

/// Команда приводу: уставки угловых скоростей колёс [рад/с].
struct WheelCommand {
    double left = 0.0;
    double right = 0.0;
};

/// Диагностика одного шага физики.
struct SubstepInfo {
    double contact_impulse = 0.0;  ///< суммарный нормальный импульс от стен [Н·с]
    std::array<double, 2> slip{};  ///< скорость проскальзывания колёс [м/с]
    int iterations = 0;            ///< число итераций решателя
};

class DiffDrive {
 public:
    /// Сколько контактов со стенами учитывается за один шаг.
    static constexpr std::size_t kMaxContacts = 8;

    /// Проверяет параметры; std::invalid_argument при нефизичных значениях.
    explicit DiffDrive(const RobotParams& params);

    const RobotParams& params() const noexcept { return params_; }

    /// Один шаг физики длительностью dt.
    /// contacts — контакты корпуса со стенами в текущей позе (см. World::collide_circle);
    /// решатель не допускает движения внутрь препятствий.
    void step(RobotState& state, WheelCommand command, std::span<const Contact> contacts, double dt,
              SubstepInfo* info = nullptr) const noexcept;

    /// Обратная кинематика: линейная v [м/с] и угловая w [рад/с] скорости → уставки колёс.
    /// Если уставка превышает max_wheel_speed, обе масштабируются одинаково —
    /// кривизна траектории сохраняется.
    WheelCommand twist_to_wheels(double v, double w) const noexcept;

    /// Предельные скорости при качении без проскальзывания.
    double max_linear_speed() const noexcept { return params_.wheel_radius * params_.max_wheel_speed; }
    double max_angular_speed() const noexcept { return 2.0 * max_linear_speed() / params_.track_width; }

 private:
    RobotParams params_;
};

}  // namespace roversim
