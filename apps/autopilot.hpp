// Простейший «водитель» для демонстраций и интеграционных тестов: едет по
// кратчайшему пути методом pure pursuit. Использует истинную позу и карту —
// это не алгоритм навигации, а эталонный оператор, с которым удобно смотреть
// на физику и сенсоры.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "roversim/simulator.hpp"

namespace roversim {

/// lookahead — расстояние до целевой точки на пути [м];
/// cruise — крейсерская скорость в долях максимальной.
inline WheelCommand pure_pursuit(const Simulator& sim, double lookahead = 0.40, double cruise = 0.7) {
    const RobotState& robot = sim.robot_state();
    const Vec2 position = robot.pose.position();
    const std::vector<Vec2> path = sim.shortest_path();

    // Целевая точка: на пути в `lookahead` метрах впереди. Путь начинается с
    // текущего положения робота, поэтому расстояние отсчитывается вдоль ломаной.
    Vec2 target = sim.goal();
    double travelled = 0.0;
    for (std::size_t i = 1; i < path.size(); ++i) {
        const double segment = distance(path[i - 1], path[i]);
        if (segment > 0.0 && travelled + segment >= lookahead) {
            target = path[i - 1] + (path[i] - path[i - 1]) * ((lookahead - travelled) / segment);
            break;
        }
        travelled += segment;
    }

    const Vec2 local = rotate_inv(target - position, std::cos(robot.pose.theta), std::sin(robot.pose.theta));
    const double bearing = std::atan2(local.y, local.x);
    const DiffDrive& drive = sim.drive();

    // Точка сбоку или сзади — разворот на месте.
    if (std::abs(bearing) > 0.9) {
        return drive.twist_to_wheels(0.0, std::copysign(0.5 * drive.max_angular_speed(), bearing));
    }
    // Иначе — дуга окружности через целевую точку: кривизна = 2·y / L².
    const double d2 = dot(local, local);
    const double curvature = d2 > 1e-9 ? 2.0 * local.y / d2 : 0.0;
    const double near_goal = std::min(1.0, 0.25 + distance(position, sim.goal()));
    const double v = cruise * drive.max_linear_speed() * std::cos(bearing) * near_goal;
    return drive.twist_to_wheels(v, curvature * v);
}

}  // namespace roversim
