#include "roversim/robot.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace roversim {
namespace {

constexpr int kMaxIterations = 32;
constexpr double kConvergence = 1e-13;  // изменение скорости за итерацию [м/с]

// Колесо с приводом на протяжении одного шага интегрирования.
//
// Вращение колеса интегрируется неявным методом Эйлера:
//     I·(w⁺ − w) = dt·clamp(K·(w_ref − w⁺), ±T) − r·J,
// где J — импульс силы тяги, которую шина прикладывает к корпусу (на обод колеса
// действует равный и противоположный). Правая часть монотонна по w⁺, поэтому
// уравнение решается в замкнутой форме и схема устойчива при любом dt — даже
// когда постоянная времени привода I/K много меньше шага.
struct Wheel {
    double w0;         // скорость в начале шага
    double w_ref;      // уставка
    double inertia;    // I
    double gain_dt;    // K·dt
    double torque_dt;  // T·dt
    double radius;     // r
    double j_lo;       // при J < j_lo привод насыщен на −T
    double j_hi;       // при J > j_hi привод насыщен на +T

    Wheel(const RobotParams& p, double dt, double start_speed, double target_speed) noexcept
        : w0(start_speed),
          w_ref(target_speed),
          inertia(p.wheel_inertia),
          gain_dt(p.motor_gain * dt),
          torque_dt(p.motor_max_torque * dt),
          radius(p.wheel_radius),
          j_lo(-std::numeric_limits<double>::infinity()),
          j_hi(std::numeric_limits<double>::infinity()) {
        if (gain_dt > 0.0) {
            const double band = torque_dt * (inertia + gain_dt) / gain_dt;
            const double base = inertia * (w_ref - w0);
            j_lo = (-band - base) / radius;
            j_hi = (band - base) / radius;
        }
    }

    /// Скорость колеса в конце шага, если за шаг шина передала корпусу импульс j.
    double speed_after(double j) const noexcept {
        const double linear = (inertia * w0 + gain_dt * w_ref - radius * j) / (inertia + gain_dt);
        const double motor = gain_dt * (w_ref - linear);  // импульс момента без насыщения
        if (motor > torque_dt) return w0 + (torque_dt - radius * j) / inertia;
        if (motor < -torque_dt) return w0 + (-torque_dt - radius * j) / inertia;
        return linear;
    }

    // Закон Кулона для пятна контакта: найти импульс J ∈ [−j_max, j_max], при
    // котором проскальзывание
    //     slip(J) = r·w⁺(J) − (contact_speed + compliance·(J − j_current))
    // равно нулю (сцепление), либо остановиться на границе (буксование/юз).
    // contact_speed — текущая скорость корпуса в точке контакта, compliance —
    // её приращение на единицу импульса. slip(J) строго убывает и кусочно-линейна
    // с изломами в j_lo, j_hi, поэтому корень находится точно, без итераций.
    double solve(double j_current, double contact_speed, double compliance, double j_max) const noexcept {
        const auto slip = [&](double j) {
            return radius * speed_after(j) - (contact_speed + compliance * (j - j_current));
        };
        double lo = -j_max, hi = j_max;
        double slip_lo = slip(lo);
        if (slip_lo <= 0.0) return lo;
        double slip_hi = slip(hi);
        if (slip_hi >= 0.0) return hi;
        for (const double breakpoint : {j_lo, j_hi}) {
            if (breakpoint > lo && breakpoint < hi) {
                const double value = slip(breakpoint);
                if (value > 0.0) {
                    lo = breakpoint;
                    slip_lo = value;
                } else {
                    hi = breakpoint;
                    slip_hi = value;
                }
            }
        }
        return lo + (hi - lo) * (slip_lo / (slip_lo - slip_hi));
    }
};

}  // namespace

DiffDrive::DiffDrive(const RobotParams& params) : params_(params) {
    const auto require = [](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(std::string("RobotParams: ") + what);
    };
    const RobotParams& p = params_;
    require(p.radius > 0.0, "radius must be > 0");
    require(p.wheel_radius > 0.0, "wheel_radius must be > 0");
    require(p.track_width > 0.0, "track_width must be > 0");
    require(p.mass > 0.0, "mass must be > 0");
    require(p.inertia_z > 0.0, "inertia_z must be > 0");
    require(p.wheel_inertia > 0.0, "wheel_inertia must be > 0");
    require(p.motor_gain >= 0.0, "motor_gain must be >= 0");
    require(p.motor_max_torque >= 0.0, "motor_max_torque must be >= 0");
    require(p.max_wheel_speed > 0.0, "max_wheel_speed must be > 0");
    require(p.mu_long >= 0.0 && p.mu_lat >= 0.0, "friction coefficients must be >= 0");
    require(p.rolling_resistance >= 0.0, "rolling_resistance must be >= 0");
    require(p.linear_drag >= 0.0 && p.angular_drag >= 0.0, "drag coefficients must be >= 0");
    require(p.gravity > 0.0, "gravity must be > 0");
}

// Шаг по времени на уровне скоростей и импульсов (схема Стюарта — Тринкла,
// как в игровых физических движках, но с точным решением для каждого колеса).
//
//   1. Предиктор: явные силы (кориолисово слагаемое, сопротивление качению, вязкость).
//   2. Связи с ограниченными импульсами, проекционный метод Гаусса — Зейделя:
//        * продольное сцепление каждой шины  |J| ≤ mu_long·N·dt;
//        * поперечное сцепление оси          |J| ≤ mu_lat·m·g·dt;
//        * непроникновение в стены           lambda ≥ 0.
//   3. Интегрирование позы по найденным скоростям (точная дуга, экспонента SE(2)).
//
// Пока импульсы не упираются в пределы, связи выполняются точно: колёса катятся
// без проскальзывания, и одометрия совпадает с истинным движением. Когда тяги
// не хватает, импульс «обрезается» — и получается буксование или занос.
void DiffDrive::step(RobotState& s, WheelCommand cmd, std::span<const Contact> contacts, double dt,
                     SubstepInfo* info) const noexcept {
    const RobotParams& p = params_;
    const double inv_m = 1.0 / p.mass;
    const double inv_iz = 1.0 / p.inertia_z;
    const double half_track = 0.5 * p.track_width;

    double vx = s.velocity.x, vy = s.velocity.y, w = s.yaw_rate;

    // ---- 1. Предиктор ----
    // Связанная СК вращается вместе с корпусом, поэтому постоянный в мировой СК
    // вектор скорости в ней поворачивается: dv/dt = −w × v.
    {
        const double vx_old = vx;
        vx += dt * w * vy;
        vy -= dt * w * vx_old;
    }
    // Сопротивление качению — сухое трение: тормозит, но не разворачивает движение.
    const double dv_roll = p.rolling_resistance * p.gravity * dt;
    vx = vx > 0.0 ? std::max(0.0, vx - dv_roll) : std::min(0.0, vx + dv_roll);
    // Вязкое сопротивление учитывается неявно: устойчиво при любом dt.
    const double linear_decay = 1.0 / (1.0 + dt * p.linear_drag * inv_m);
    vx *= linear_decay;
    vy *= linear_decay;
    w *= 1.0 / (1.0 + dt * p.angular_drag * inv_iz);

    // ---- 2. Связи ----
    const Wheel left(p, dt, s.wheel_speed[0], std::clamp(cmd.left, -p.max_wheel_speed, p.max_wheel_speed));
    const Wheel right(p, dt, s.wheel_speed[1], std::clamp(cmd.right, -p.max_wheel_speed, p.max_wheel_speed));
    // Податливость корпуса в точке контакта колеса: приращение скорости на единицу импульса.
    const double wheel_compliance = inv_m + half_track * half_track * inv_iz;
    const double j_long_max = p.mu_long * 0.5 * p.mass * p.gravity * dt;  // вес поровну на два колеса
    const double j_lat_max = p.mu_lat * p.mass * p.gravity * dt;

    // Нормали контактов в связанной СК. Контакт «упреждающий»: при зазоре gap
    // сближаться с препятствием можно не быстрее gap/dt — тогда к концу шага
    // зазор не станет отрицательным и корпус не войдёт в стену.
    const double cos_t = std::cos(s.pose.theta), sin_t = std::sin(s.pose.theta);
    struct BodyContact {
        double nx, ny, min_speed, impulse;
    };
    BodyContact body_contacts[kMaxContacts];
    const std::size_t num_contacts = std::min(contacts.size(), kMaxContacts);
    for (std::size_t i = 0; i < num_contacts; ++i) {
        const Vec2 n = rotate_inv(contacts[i].normal, cos_t, sin_t);
        body_contacts[i] = {n.x, n.y, -std::max(contacts[i].gap, 0.0) / dt, 0.0};
    }

    double j_left = 0.0, j_right = 0.0, j_lat = 0.0;
    int iterations = 0;
    while (iterations < kMaxIterations) {
        ++iterations;
        double change = 0.0;
        {  // Левое колесо (y = +half_track): скорость точки контакта vx − half_track·w.
            const double j = left.solve(j_left, vx - half_track * w, wheel_compliance, j_long_max);
            const double delta = j - j_left;
            j_left = j;
            vx += delta * inv_m;
            w -= delta * half_track * inv_iz;
            change = std::max(change, std::abs(delta) * inv_m);
        }
        {  // Правое колесо (y = −half_track): скорость точки контакта vx + half_track·w.
            const double j = right.solve(j_right, vx + half_track * w, wheel_compliance, j_long_max);
            const double delta = j - j_right;
            j_right = j;
            vx += delta * inv_m;
            w += delta * half_track * inv_iz;
            change = std::max(change, std::abs(delta) * inv_m);
        }
        {  // Поперечное сцепление: ось не скользит вбок, пока хватает трения.
            const double j = std::clamp(j_lat - p.mass * vy, -j_lat_max, j_lat_max);
            const double delta = j - j_lat;
            j_lat = j;
            vy += delta * inv_m;
            change = std::max(change, std::abs(delta) * inv_m);
        }
        // Стены решаются последними: условие непроникновения выполняется точнее всего.
        for (std::size_t i = 0; i < num_contacts; ++i) {
            BodyContact& c = body_contacts[i];
            const double normal_speed = c.nx * vx + c.ny * vy;
            const double impulse = std::max(0.0, c.impulse + p.mass * (c.min_speed - normal_speed));
            const double delta = impulse - c.impulse;
            c.impulse = impulse;
            vx += c.nx * delta * inv_m;
            vy += c.ny * delta * inv_m;
            change = std::max(change, std::abs(delta) * inv_m);
        }
        if (change < kConvergence) break;
    }

    // ---- 3. Обновление состояния ----
    s.wheel_speed = {left.speed_after(j_left), right.speed_after(j_right)};
    s.wheel_angle[0] += dt * s.wheel_speed[0];
    s.wheel_angle[1] += dt * s.wheel_speed[1];
    s.velocity = {vx, vy};
    s.yaw_rate = w;

    // Поза интегрируется точно для постоянных за шаг скоростей в связанной СК:
    // движение по дуге окружности, а не по хорде.
    const double phi = w * dt;
    double sinc = 0.0, cosc = 0.0;  // sin(phi)/phi и (1 − cos(phi))/phi
    if (std::abs(phi) < 1e-4) {
        const double phi2 = phi * phi;
        sinc = 1.0 - phi2 / 6.0;
        cosc = phi * (0.5 - phi2 / 24.0);
    } else {
        sinc = std::sin(phi) / phi;
        cosc = (1.0 - std::cos(phi)) / phi;
    }
    const double dx_body = dt * (vx * sinc - vy * cosc);
    const double dy_body = dt * (vx * cosc + vy * sinc);
    s.pose.x += cos_t * dx_body - sin_t * dy_body;
    s.pose.y += sin_t * dx_body + cos_t * dy_body;
    s.pose.theta = wrap_angle(s.pose.theta + phi);

    if (info != nullptr) {
        info->contact_impulse = 0.0;
        for (std::size_t i = 0; i < num_contacts; ++i) info->contact_impulse += body_contacts[i].impulse;
        info->slip = {p.wheel_radius * s.wheel_speed[0] - (vx - half_track * w),
                      p.wheel_radius * s.wheel_speed[1] - (vx + half_track * w)};
        info->iterations = iterations;
    }
}

WheelCommand DiffDrive::twist_to_wheels(double v, double w) const noexcept {
    const double half_track = 0.5 * params_.track_width;
    double left = (v - w * half_track) / params_.wheel_radius;
    double right = (v + w * half_track) / params_.wheel_radius;
    const double peak = std::max(std::abs(left), std::abs(right));
    if (peak > params_.max_wheel_speed) {
        const double scale = params_.max_wheel_speed / peak;
        left *= scale;
        right *= scale;
    }
    return {left, right};
}

}  // namespace roversim
