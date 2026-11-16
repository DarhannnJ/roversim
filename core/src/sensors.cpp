#include "roversim/sensors.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace roversim {

// ------------------------------------------------------------------- лидар

Lidar2D::Lidar2D(const LidarConfig& config) : cfg_(config) {
    if (cfg_.num_rays < 1) throw std::invalid_argument("LidarConfig: num_rays must be >= 1");
    if (!(cfg_.fov > 0.0) || cfg_.fov > kTwoPi + 1e-9) {
        throw std::invalid_argument("LidarConfig: fov must be in (0, 2*pi]");
    }
    if (!(cfg_.range_min >= 0.0) || !(cfg_.range_max > cfg_.range_min)) {
        throw std::invalid_argument("LidarConfig: need 0 <= range_min < range_max");
    }
    if (cfg_.noise_std < 0.0 || cfg_.noise_std_rel < 0.0 || cfg_.resolution < 0.0 ||
        cfg_.dropout_prob < 0.0 || cfg_.dropout_prob > 1.0) {
        throw std::invalid_argument("LidarConfig: noise parameters out of range");
    }

    const bool full_circle = cfg_.fov >= kTwoPi - 1e-9;
    angle_min_ = -0.5 * cfg_.fov;
    if (cfg_.num_rays > 1) {
        angle_inc_ = full_circle ? cfg_.fov / cfg_.num_rays : cfg_.fov / (cfg_.num_rays - 1);
    }
    // Тригонометрия считается один раз здесь; в scan() направление каждого луча
    // получается поворотом на курс — четыре умножения вместо sin/cos на луч.
    period_ = static_cast<std::size_t>(cfg_.num_rays);
    if (angle_inc_ > 0.0) period_ = std::max(period_, static_cast<std::size_t>(std::llround(kTwoPi / angle_inc_)));
    dirs_.reserve(static_cast<std::size_t>(cfg_.num_rays));
    for (int i = 0; i < cfg_.num_rays; ++i) {
        const double a = cfg_.mount.theta + angle_min_ + i * angle_inc_;
        dirs_.push_back({std::cos(a), std::sin(a)});
    }
}

void Lidar2D::scan(const World& world, const Pose2& body, std::span<const Circle> moving, Rng& rng,
                   std::span<float> out) const noexcept {
    const double c = std::cos(body.theta), s = std::sin(body.theta);
    const Vec2 origin = body.position() + rotate({cfg_.mount.x, cfg_.mount.y}, c, s);
    world.raycast_scan(origin, c, s, dirs_, cfg_.range_max, out);
    const std::size_t n = std::min(out.size(), dirs_.size());

    // Подвижные препятствия. Круг радиуса R на расстоянии d закрывает на скане
    // сектор полушириной asin(R/d), поэтому проверяются только лучи этого сектора,
    // а не все подряд.
    for (const Circle& obstacle : moving) {
        const Vec2 rel = obstacle.center - origin;
        const double dist = norm(rel);
        if (dist - obstacle.radius >= cfg_.range_max) continue;
        std::size_t first = 0, count = period_;  // сенсор внутри круга или луч один: проверяем всё
        if (dist > obstacle.radius && angle_inc_ > 0.0) {
            const double half = std::asin(obstacle.radius / dist) + angle_inc_;  // плюс луч запаса
            const double center = std::atan2(rel.y, rel.x) - body.theta - cfg_.mount.theta;
            double offset = center - half - angle_min_;       // угол от первого луча до начала сектора
            offset -= kTwoPi * std::floor(offset / kTwoPi);   // приводим к [0, 2*pi)
            first = static_cast<std::size_t>(offset / angle_inc_);
            count = std::min(period_, static_cast<std::size_t>(2.0 * half / angle_inc_) + 2);
        }
        for (std::size_t k = 0; k < count; ++k) {
            const std::size_t i = (first + k) % period_;
            if (i >= n) continue;  // направление вне угла обзора
            const Vec2 dir = rotate(dirs_[i], c, s);
            out[i] = static_cast<float>(ray_circle_distance(origin, dir, obstacle, out[i]));
        }
    }

    const double inv_resolution = cfg_.resolution > 0.0 ? 1.0 / cfg_.resolution : 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        // Случайные числа берутся на каждый луч безусловно — см. комментарий в заголовке.
        const double z = rng.normal();
        const bool dropped = cfg_.dropout_prob > 0.0 && rng.uniform() < cfg_.dropout_prob;
        double r = out[i];
        if (r >= cfg_.range_max || dropped) {
            r = cfg_.range_max;  // отражения нет
        } else {
            r = std::clamp(r + (cfg_.noise_std + cfg_.noise_std_rel * r) * z, cfg_.range_min, cfg_.range_max);
            if (inv_resolution > 0.0) {
                // Округление до шага квантования через целое: значение неотрицательно,
                // поэтому отбрасывание дробной части (x + 0.5) равно округлению.
                const auto ticks = static_cast<std::int64_t>(r * inv_resolution + 0.5);
                r = std::min(static_cast<double>(ticks) * cfg_.resolution, cfg_.range_max);
            }
        }
        out[i] = static_cast<float>(r);
    }
}

// --------------------------------------------------------------------- IMU

Imu::Imu(const ImuConfig& config, double nominal_dt) : cfg_(config), nominal_dt_(nominal_dt) {}

void Imu::reset(const RobotState& state, Rng& rng) noexcept {
    gyro_bias_ = cfg_.gyro_bias_init * rng.normal();
    accel_bias_ = {cfg_.accel_bias_init * rng.normal(), cfg_.accel_bias_init * rng.normal()};
    start_velocity_ = state.velocity;
    coriolis_ = {};
    delta_yaw_ = 0.0;
    elapsed_ = 0.0;
}

void Imu::accumulate(const RobotState& state, double dt) noexcept {
    // Кажущееся ускорение в связанной СК: f = dv/dt + w × v. Первое слагаемое
    // за интервал даёт разность скоростей, второе интегрируется здесь.
    coriolis_ += Vec2{-state.yaw_rate * state.velocity.y, state.yaw_rate * state.velocity.x} * dt;
    delta_yaw_ += state.yaw_rate * dt;
    elapsed_ += dt;
}

ImuSample Imu::sample(const RobotState& state, Rng& rng) noexcept {
    const bool moved = elapsed_ > 0.0;
    const double interval = moved ? elapsed_ : nominal_dt_;
    const double inv_t = 1.0 / interval;

    // Смещения — винеровский процесс: дисперсия приращения пропорциональна времени.
    const double sqrt_t = std::sqrt(interval);
    gyro_bias_ += cfg_.gyro_bias_walk * sqrt_t * rng.normal();
    accel_bias_.x += cfg_.accel_bias_walk * sqrt_t * rng.normal();
    accel_bias_.y += cfg_.accel_bias_walk * sqrt_t * rng.normal();

    // Белый шум с плотностью n после усреднения за T имеет СКО n/√T.
    const double white = std::sqrt(inv_t);
    ImuSample out;
    const double gyro_true = moved ? delta_yaw_ * inv_t : 0.0;
    const Vec2 accel_true = moved ? (state.velocity - start_velocity_ + coriolis_) * inv_t : Vec2{};
    out.gyro_z = gyro_true + gyro_bias_ + cfg_.gyro_noise_density * white * rng.normal();
    out.accel_x = accel_true.x + accel_bias_.x + cfg_.accel_noise_density * white * rng.normal();
    out.accel_y = accel_true.y + accel_bias_.y + cfg_.accel_noise_density * white * rng.normal();

    start_velocity_ = state.velocity;
    coriolis_ = {};
    delta_yaw_ = 0.0;
    elapsed_ = 0.0;
    return out;
}

// -------------------------------------------------------------- одометрия

WheelOdometry::WheelOdometry(const OdomConfig& config, const RobotParams& robot)
    : cfg_(config), wheel_radius_(robot.wheel_radius), track_width_(robot.track_width) {
    if (cfg_.ticks_per_rev < 0) throw std::invalid_argument("OdomConfig: ticks_per_rev must be >= 0");
    ticks_per_rad_ = cfg_.ticks_per_rev / kTwoPi;
    radius_assumed_ = {wheel_radius_, wheel_radius_};
    track_assumed_ = track_width_;
}

void WheelOdometry::reset(const RobotState& state, Rng& rng) noexcept {
    // Ошибка калибровки постоянна в пределах эпизода — это систематический дрейф.
    radius_assumed_[0] = wheel_radius_ * (1.0 + cfg_.wheel_radius_error * rng.normal());
    radius_assumed_[1] = wheel_radius_ * (1.0 + cfg_.wheel_radius_error * rng.normal());
    track_assumed_ = track_width_ * (1.0 + cfg_.track_width_error * rng.normal());
    true_angle_ = state.wheel_angle;
    measured_angle_ = {0.0, 0.0};
    ticks_ = {0, 0};
    sample_ = {};
    sample_.pose = state.pose;
}

const OdomSample& WheelOdometry::update(const RobotState& state, double dt, Rng& rng) noexcept {
    std::array<double, 2> delta{};
    for (std::size_t i = 0; i < 2; ++i) {
        const double true_delta = state.wheel_angle[i] - true_angle_[i];
        true_angle_[i] = state.wheel_angle[i];
        // Дисперсия шума пропорциональна пройденному углу: ошибка копится как
        // случайное блуждание и не зависит от того, на сколько шагов разбит путь.
        measured_angle_[i] += true_delta + cfg_.slip_noise * std::sqrt(std::abs(true_delta)) * rng.normal();
        if (ticks_per_rad_ > 0.0) {
            // Энкодер считает целые тики; дробный остаток не теряется, а ждёт следующего шага.
            const auto ticks = static_cast<std::int64_t>(std::floor(measured_angle_[i] * ticks_per_rad_));
            delta[i] = static_cast<double>(ticks - ticks_[i]) / ticks_per_rad_;
            ticks_[i] = ticks;
        } else {
            delta[i] = measured_angle_[i];
            measured_angle_[i] = 0.0;
        }
    }

    const double left = radius_assumed_[0] * delta[0];
    const double right = radius_assumed_[1] * delta[1];
    const double distance = 0.5 * (left + right);
    const double turn = (right - left) / track_assumed_;
    // Интегрирование по средней точке (второй порядок точности).
    const double heading = sample_.pose.theta + 0.5 * turn;
    sample_.pose.x += distance * std::cos(heading);
    sample_.pose.y += distance * std::sin(heading);
    sample_.pose.theta = wrap_angle(sample_.pose.theta + turn);

    const double inv_dt = dt > 0.0 ? 1.0 / dt : 0.0;
    sample_.wheel_velocity = {delta[0] * inv_dt, delta[1] * inv_dt};
    sample_.v = distance * inv_dt;
    sample_.w = turn * inv_dt;
    return sample_;
}

}  // namespace roversim
