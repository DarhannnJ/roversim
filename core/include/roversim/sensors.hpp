// Модели сенсоров: 2D-лидар, IMU, колёсная одометрия.
//
// Каждый сенсор получает собственный поток ГСЧ (см. Rng::stream) и за одно
// измерение потребляет фиксированное число случайных чисел, не зависящее от
// сцены. Поэтому шум воспроизводим и одинаков в прогонах с разным управлением —
// удобно для парных сравнений алгоритмов.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "roversim/math.hpp"
#include "roversim/rng.hpp"
#include "roversim/robot.hpp"
#include "roversim/world.hpp"

namespace roversim {

// ------------------------------------------------------------------- лидар

struct LidarConfig {
    int num_rays = 360;         ///< число лучей в скане
    double fov = kTwoPi;        ///< угол обзора [рад], симметрично относительно оси сенсора
    double range_min = 0.05;    ///< ближняя граница [м]
    double range_max = 10.0;    ///< дальняя граница [м]; её же возвращает луч без отражения
    double noise_std = 0.01;    ///< СКО шума дальности, постоянная часть [м]
    double noise_std_rel = 0.0; ///< СКО шума, растущая с дальностью [м/м]
    double dropout_prob = 0.0;  ///< вероятность потери отражения
    double resolution = 0.001;  ///< шаг квантования дальности [м]; 0 — без квантования
    Pose2 mount{};              ///< поза сенсора в связанной СК робота
};

class Lidar2D {
 public:
    explicit Lidar2D(const LidarConfig& config);

    const LidarConfig& config() const noexcept { return cfg_; }
    int num_rays() const noexcept { return cfg_.num_rays; }
    /// Угол луча i в СК сенсора: angle_min + i * angle_increment.
    /// Полный круг делится на num_rays равных секторов; при неполном обзоре
    /// первый и последний лучи идут точно по краям fov.
    double angle_min() const noexcept { return angle_min_; }
    double angle_increment() const noexcept { return angle_inc_; }

    /// Снимает скан из позы робота `body` и записывает num_rays дальностей [м] в out.
    /// `moving` — подвижные препятствия (круги), которых нет в статическом мире.
    void scan(const World& world, const Pose2& body, std::span<const Circle> moving, Rng& rng,
              std::span<float> out) const noexcept;

 private:
    LidarConfig cfg_;
    double angle_min_ = 0.0;
    double angle_inc_ = 0.0;
    std::size_t period_ = 1;  ///< число шагов angle_inc_ в полном обороте
    std::vector<Vec2> dirs_;  ///< единичные направления лучей в связанной СК робота
};

// --------------------------------------------------------------------- IMU

/// Параметры шума заданы как спектральные плотности, поэтому не зависят от
/// частоты опроса: при смене шага управления статистика остаётся физически верной.
struct ImuConfig {
    double gyro_noise_density = 2e-4;   ///< белый шум гироскопа [рад/с/√Гц]
    double gyro_bias_walk = 2e-5;       ///< случайное блуждание смещения гироскопа [рад/с²/√Гц]
    double gyro_bias_init = 2e-3;       ///< СКО начального смещения гироскопа [рад/с]
    double accel_noise_density = 2e-3;  ///< белый шум акселерометра [м/с²/√Гц]
    double accel_bias_walk = 2e-4;      ///< случайное блуждание смещения акселерометра [м/с³/√Гц]
    double accel_bias_init = 2e-2;      ///< СКО начального смещения акселерометра [м/с²]
};

/// Показания, усреднённые за шаг управления (предынтегрированное измерение).
struct ImuSample {
    double gyro_z = 0.0;   ///< угловая скорость рыскания [рад/с]
    double accel_x = 0.0;  ///< кажущееся ускорение вдоль оси робота [м/с²]
    double accel_y = 0.0;  ///< кажущееся ускорение влево [м/с²]
};

class Imu {
 public:
    /// nominal_dt — длительность шага управления; нужна только для первого
    /// измерения после reset, когда робот ещё не двигался.
    Imu(const ImuConfig& config, double nominal_dt);

    /// Начало эпизода: разыгрывает начальные смещения.
    void reset(const RobotState& state, Rng& rng) noexcept;
    /// Вызывается после каждого шага физики: накапливает истинное движение.
    void accumulate(const RobotState& state, double dt) noexcept;
    /// Раз в шаг управления: среднее за интервал плюс шум и смещение.
    ImuSample sample(const RobotState& state, Rng& rng) noexcept;

 private:
    ImuConfig cfg_;
    double nominal_dt_;
    double gyro_bias_ = 0.0;
    Vec2 accel_bias_{};
    Vec2 start_velocity_{};  ///< скорость в связанной СК в начале интервала
    Vec2 coriolis_{};        ///< ∫ w × v dt за интервал
    double delta_yaw_ = 0.0; ///< ∫ w dt за интервал
    double elapsed_ = 0.0;
};

// -------------------------------------------------------------- одометрия

struct OdomConfig {
    int ticks_per_rev = 4096;           ///< разрешение энкодера [тиков на оборот колеса]; 0 — без квантования
    double slip_noise = 0.002;          ///< случайная ошибка: СКО = slip_noise·sqrt(|Δφ|) [рад^0.5]
    double wheel_radius_error = 0.003;  ///< СКО относительной ошибки калибровки радиуса колеса
    double track_width_error = 0.005;   ///< СКО относительной ошибки калибровки колеи
};

struct OdomSample {
    std::array<double, 2> wheel_velocity{};  ///< скорости колёс по энкодерам [рад/с]
    double v = 0.0;                          ///< оценка линейной скорости [м/с]
    double w = 0.0;                          ///< оценка угловой скорости [рад/с]
    Pose2 pose{};                            ///< счисленная поза
};

/// Счисление пути по энкодерам — так, как это делает прошивка робота.
/// Источники ошибок: квантование, случайный шум, систематическая ошибка
/// калибровки (разыгрывается на эпизод) и настоящее проскальзывание колёс,
/// которое приходит из физики: буксующее колесо «наматывает» путь, которого не было.
class WheelOdometry {
 public:
    WheelOdometry(const OdomConfig& config, const RobotParams& robot);

    /// Начало эпизода: счисление стартует из истинной позы.
    void reset(const RobotState& state, Rng& rng) noexcept;
    /// Раз в шаг управления длительностью dt.
    const OdomSample& update(const RobotState& state, double dt, Rng& rng) noexcept;
    const OdomSample& last() const noexcept { return sample_; }

 private:
    OdomConfig cfg_;
    double wheel_radius_;
    double track_width_;
    double ticks_per_rad_ = 0.0;
    std::array<double, 2> radius_assumed_{};    ///< радиусы колёс, которые «знает» прошивка
    double track_assumed_ = 0.0;
    std::array<double, 2> true_angle_{};        ///< истинные углы колёс на прошлом шаге
    std::array<double, 2> measured_angle_{};    ///< накопленный угол с шумом
    std::array<std::int64_t, 2> ticks_{};       ///< показания счётчиков энкодеров
    OdomSample sample_{};
};

}  // namespace roversim
