// Базовая 2D-математика: векторы, позы, углы. Header-only, без зависимостей.
#pragma once

#include <cmath>
#include <numbers>

namespace roversim {

inline constexpr double kPi = std::numbers::pi;
inline constexpr double kTwoPi = 2.0 * std::numbers::pi;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    constexpr Vec2 operator+(Vec2 o) const noexcept { return {x + o.x, y + o.y}; }
    constexpr Vec2 operator-(Vec2 o) const noexcept { return {x - o.x, y - o.y}; }
    constexpr Vec2 operator*(double s) const noexcept { return {x * s, y * s}; }
    constexpr Vec2& operator+=(Vec2 o) noexcept {
        x += o.x;
        y += o.y;
        return *this;
    }
};

constexpr double dot(Vec2 a, Vec2 b) noexcept { return a.x * b.x + a.y * b.y; }
// sqrt корректно округляется по IEEE-754, поэтому (в отличие от hypot) результат
// одинаков на любой платформе.
inline double norm(Vec2 a) noexcept { return std::sqrt(dot(a, a)); }
inline double distance(Vec2 a, Vec2 b) noexcept { return norm(a - b); }

/// Поворот вектора на угол, заданный заранее вычисленными (cos, sin).
constexpr Vec2 rotate(Vec2 v, double c, double s) noexcept {
    return {c * v.x - s * v.y, s * v.x + c * v.y};
}
/// Обратный поворот (на минус тот же угол).
constexpr Vec2 rotate_inv(Vec2 v, double c, double s) noexcept {
    return {c * v.x + s * v.y, -s * v.x + c * v.y};
}

/// Поза на плоскости: положение [м] и курс [рад], отсчёт против часовой стрелки от оси X.
struct Pose2 {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;

    constexpr Vec2 position() const noexcept { return {x, y}; }
};

/// Приводит угол к [-pi, pi]. std::remainder — точная операция IEEE-754,
/// поэтому результат не зависит от реализации libm.
inline double wrap_angle(double a) noexcept { return std::remainder(a, kTwoPi); }

}  // namespace roversim
