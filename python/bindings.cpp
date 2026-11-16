// Python-биндинги ядра (модуль roversim._core).
//
// Главный принцип — наблюдения не копируются: свойства lidar / proprio / state
// возвращают numpy-массивы, которые смотрят прямо в память симулятора (или в
// разделяемую память у ShmClient). Такой массив только для чтения, удерживает
// своего владельца от удаления, а его содержимое меняется при следующем
// reset()/step(). Тот же массив без копирования превращается в тензор:
// torch.from_numpy(sim.lidar).
//
// На время reset()/step() GIL отпускается, поэтому несколько симуляторов можно
// шагать из разных потоков Python по-настоящему параллельно.
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "roversim/nav/navigator.hpp"
#include "roversim/shm.hpp"
#include "roversim/vec_simulator.hpp"
#include "roversim/viz.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace roversim;

namespace {

using Pair = std::array<double, 2>;
using Triple = std::array<double, 3>;

/// numpy-массив поверх памяти C++ без копирования. `owner` — объект Python,
/// которому принадлежит память: массив держит на него ссылку.
template <typename T>
py::array_t<T> view(const T* data, std::vector<py::ssize_t> shape, py::handle owner) {
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = static_cast<py::ssize_t>(sizeof(T));
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    py::array_t<T> array(std::move(shape), std::move(strides), data, owner);
    // Только чтение: буфер принадлежит симулятору, случайная запись испортила бы наблюдение.
    py::detail::array_proxy(array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return array;
}

template <typename T>
py::array_t<T> view(std::span<const T> data, py::handle owner) {
    return view(data.data(), {static_cast<py::ssize_t>(data.size())}, owner);
}

/// Флаги хранятся как uint8 со значениями 0/1 — это же представление numpy.bool_.
py::array_t<bool> flag_view(std::span<const std::uint8_t> data, py::handle owner) {
    return view(reinterpret_cast<const bool*>(data.data()), {static_cast<py::ssize_t>(data.size())}, owner);
}

py::array_t<double> lidar_angles(double angle_min, double increment, std::size_t count) {
    py::array_t<double> angles(static_cast<py::ssize_t>(count));
    auto out = angles.mutable_unchecked<1>();
    for (std::size_t i = 0; i < count; ++i) out(static_cast<py::ssize_t>(i)) = angle_min + static_cast<double>(i) * increment;
    return angles;
}

std::shared_ptr<const World> finalized(const std::shared_ptr<World>& world) {
    if (!world) throw py::value_error("world is None");
    if (!world->finalized()) world->finalize();
    return world;
}

/// Один мир или последовательность миров → список для VecSimulator.
std::vector<std::shared_ptr<const World>> world_list(const py::object& worlds) {
    std::vector<std::shared_ptr<const World>> list;
    if (py::isinstance<World>(worlds)) {
        list.push_back(finalized(worlds.cast<std::shared_ptr<World>>()));
    } else {
        for (const py::handle item : worlds) list.push_back(finalized(item.cast<std::shared_ptr<World>>()));
    }
    return list;
}

/// Общая часть reset(): либо случайные старт и цель, либо заданные явно.
template <typename Target>
void reset_any(Target& target, std::uint64_t seed, const std::optional<Triple>& start, const std::optional<Pair>& goal) {
    if (start.has_value() != goal.has_value()) throw py::value_error("pass both start and goal, or neither");
    py::gil_scoped_release release;
    if (start) {
        target.reset(seed, Pose2{(*start)[0], (*start)[1], (*start)[2]}, Vec2{(*goal)[0], (*goal)[1]});
    } else {
        target.reset(seed);
    }
}

std::string repr(const StepResult& r) {
    return "StepResult(reward=" + std::to_string(r.reward) + ", terminated=" + (r.terminated ? "True" : "False") +
           ", truncated=" + (r.truncated ? "True" : "False") + ", collision=" + (r.collision ? "True" : "False") +
           ", success=" + (r.success ? "True" : "False") + ")";
}

void bind_config(py::module_& m) {
    py::class_<RobotParams>(m, "RobotParams", "Геометрия, инерция, привод и трение робота.")
        .def(py::init<>())
        .def_readwrite("radius", &RobotParams::radius, "радиус корпуса (круг столкновений) [м]")
        .def_readwrite("wheel_radius", &RobotParams::wheel_radius, "радиус ведущих колёс [м]")
        .def_readwrite("track_width", &RobotParams::track_width, "колея [м]")
        .def_readwrite("mass", &RobotParams::mass, "масса [кг]")
        .def_readwrite("inertia_z", &RobotParams::inertia_z, "момент инерции вокруг вертикали [кг·м²]")
        .def_readwrite("wheel_inertia", &RobotParams::wheel_inertia, "колесо + приведённый ротор [кг·м²]")
        .def_readwrite("motor_gain", &RobotParams::motor_gain, "жёсткость привода скорости [Н·м·с/рад]")
        .def_readwrite("motor_max_torque", &RobotParams::motor_max_torque, "предельный момент на колесе [Н·м]")
        .def_readwrite("max_wheel_speed", &RobotParams::max_wheel_speed, "ограничение уставки колеса [рад/с]")
        .def_readwrite("mu_long", &RobotParams::mu_long, "продольное трение шин")
        .def_readwrite("mu_lat", &RobotParams::mu_lat, "поперечное трение шин")
        .def_readwrite("rolling_resistance", &RobotParams::rolling_resistance, "коэффициент сопротивления качению")
        .def_readwrite("linear_drag", &RobotParams::linear_drag, "вязкое сопротивление [Н·с/м]")
        .def_readwrite("angular_drag", &RobotParams::angular_drag, "вязкое сопротивление вращению [Н·м·с/рад]")
        .def_readwrite("gravity", &RobotParams::gravity, "[м/с²]");

    py::class_<LidarConfig>(m, "LidarConfig", "Параметры 2D-лидара.")
        .def(py::init<>())
        .def_readwrite("num_rays", &LidarConfig::num_rays, "число лучей в скане")
        .def_readwrite("fov", &LidarConfig::fov, "угол обзора [рад]")
        .def_readwrite("range_min", &LidarConfig::range_min, "ближняя граница [м]")
        .def_readwrite("range_max", &LidarConfig::range_max, "дальняя граница [м]")
        .def_readwrite("noise_std", &LidarConfig::noise_std, "СКО шума дальности [м]")
        .def_readwrite("noise_std_rel", &LidarConfig::noise_std_rel, "СКО шума, растущая с дальностью [м/м]")
        .def_readwrite("dropout_prob", &LidarConfig::dropout_prob, "вероятность потери отражения")
        .def_readwrite("resolution", &LidarConfig::resolution, "шаг квантования дальности [м]; 0 — нет")
        .def_property(
            "mount", [](const LidarConfig& c) { return Triple{c.mount.x, c.mount.y, c.mount.theta}; },
            [](LidarConfig& c, const Triple& v) { c.mount = {v[0], v[1], v[2]}; },
            "поза сенсора в связанной СК робота: (x, y, theta)");

    py::class_<ImuConfig>(m, "ImuConfig", "Шумы IMU, заданные спектральными плотностями.")
        .def(py::init<>())
        .def_readwrite("gyro_noise_density", &ImuConfig::gyro_noise_density, "[рад/с/√Гц]")
        .def_readwrite("gyro_bias_walk", &ImuConfig::gyro_bias_walk, "[рад/с²/√Гц]")
        .def_readwrite("gyro_bias_init", &ImuConfig::gyro_bias_init, "СКО начального смещения [рад/с]")
        .def_readwrite("accel_noise_density", &ImuConfig::accel_noise_density, "[м/с²/√Гц]")
        .def_readwrite("accel_bias_walk", &ImuConfig::accel_bias_walk, "[м/с³/√Гц]")
        .def_readwrite("accel_bias_init", &ImuConfig::accel_bias_init, "СКО начального смещения [м/с²]");

    py::class_<OdomConfig>(m, "OdomConfig", "Ошибки колёсной одометрии.")
        .def(py::init<>())
        .def_readwrite("ticks_per_rev", &OdomConfig::ticks_per_rev, "тиков энкодера на оборот колеса; 0 — без квантования")
        .def_readwrite("slip_noise", &OdomConfig::slip_noise, "случайная ошибка: СКО = slip_noise·sqrt(|Δφ|)")
        .def_readwrite("wheel_radius_error", &OdomConfig::wheel_radius_error, "СКО ошибки калибровки радиуса колеса")
        .def_readwrite("track_width_error", &OdomConfig::track_width_error, "СКО ошибки калибровки колеи");

    py::class_<TaskConfig>(m, "TaskConfig", "Задача «доехать до точки» и её награда.")
        .def(py::init<>())
        .def_readwrite("goal_radius", &TaskConfig::goal_radius, "радиус достижения цели [м]")
        .def_readwrite("max_steps", &TaskConfig::max_steps, "лимит шагов эпизода; 0 — без лимита")
        .def_readwrite("terminate_on_collision", &TaskConfig::terminate_on_collision, "завершать эпизод при касании стены")
        .def_readwrite("min_goal_distance", &TaskConfig::min_goal_distance, "минимальная дистанция до случайной цели [м]")
        .def_readwrite("spawn_clearance", &TaskConfig::spawn_clearance, "запас до стен при выборе старта и цели [м]")
        .def_readwrite("geodesic", &TaskConfig::geodesic, "прогресс по кратчайшему пути, а не по прямой")
        .def_readwrite("path_hint", &TaskConfig::path_hint, "давать в наблюдении направление по маршруту (PATH_COS/PATH_SIN)")
        .def_readwrite("lookahead", &TaskConfig::lookahead, "расстояние до точки подсказки вдоль маршрута [м]")
        .def_readwrite("progress_weight", &TaskConfig::progress_weight, "вес прогресса к цели [награда/м]")
        .def_readwrite("step_penalty", &TaskConfig::step_penalty, "штраф за шаг")
        .def_readwrite("collision_penalty", &TaskConfig::collision_penalty, "штраф за касание стены")
        .def_readwrite("success_reward", &TaskConfig::success_reward, "награда за достижение цели");

    py::class_<ObstacleConfig>(m, "ObstacleConfig", "Подвижные препятствия: круги, отражающиеся от стен.")
        .def(py::init<>())
        .def_readwrite("count", &ObstacleConfig::count, "сколько препятствий создавать в каждом эпизоде")
        .def_readwrite("radius", &ObstacleConfig::radius, "радиус [м]")
        .def_readwrite("speed_min", &ObstacleConfig::speed_min, "минимальная скорость [м/с]")
        .def_readwrite("speed_max", &ObstacleConfig::speed_max, "максимальная скорость [м/с]");

    py::class_<SimConfig>(m, "SimConfig", "Полная конфигурация симулятора.")
        .def(py::init<>())
        .def_readwrite("physics_dt", &SimConfig::physics_dt, "шаг физики [с]")
        .def_readwrite("substeps", &SimConfig::substeps, "шагов физики на один step()")
        .def_readwrite("robot", &SimConfig::robot)
        .def_readwrite("lidar", &SimConfig::lidar)
        .def_readwrite("imu", &SimConfig::imu)
        .def_readwrite("odom", &SimConfig::odom)
        .def_readwrite("task", &SimConfig::task)
        .def_readwrite("obstacles", &SimConfig::obstacles)
        .def_property_readonly("control_dt", &SimConfig::control_dt, "период управления [с]")
        .def("copy", [](const SimConfig& c) { return c; }, "Независимая копия конфигурации.");

    py::class_<StepResult>(m, "StepResult", "Итог шага: награда и флаги завершения.")
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("terminated", &StepResult::terminated, "эпизод завершён: цель или столкновение")
        .def_readonly("truncated", &StepResult::truncated, "эпизод прерван по лимиту шагов")
        .def_readonly("collision", &StepResult::collision, "на этом шаге робот касался препятствия")
        .def_readonly("success", &StepResult::success, "цель достигнута")
        .def("__repr__", &repr);

    // Индексы векторов наблюдения: sim.proprio[roversim.obs.GOAL_DISTANCE].
    py::module_ obs_module = m.def_submodule("obs", "Индексы вектора proprio (float32).");
    obs_module.attr("GYRO_Z") = static_cast<int>(obs::GyroZ);
    obs_module.attr("ACCEL_X") = static_cast<int>(obs::AccelX);
    obs_module.attr("ACCEL_Y") = static_cast<int>(obs::AccelY);
    obs_module.attr("WHEEL_VEL_LEFT") = static_cast<int>(obs::WheelVelLeft);
    obs_module.attr("WHEEL_VEL_RIGHT") = static_cast<int>(obs::WheelVelRight);
    obs_module.attr("ODOM_V") = static_cast<int>(obs::OdomV);
    obs_module.attr("ODOM_W") = static_cast<int>(obs::OdomW);
    obs_module.attr("ODOM_X") = static_cast<int>(obs::OdomX);
    obs_module.attr("ODOM_Y") = static_cast<int>(obs::OdomY);
    obs_module.attr("ODOM_THETA") = static_cast<int>(obs::OdomTheta);
    obs_module.attr("GOAL_DISTANCE") = static_cast<int>(obs::GoalDistance);
    obs_module.attr("GOAL_COS") = static_cast<int>(obs::GoalCos);
    obs_module.attr("GOAL_SIN") = static_cast<int>(obs::GoalSin);
    obs_module.attr("PATH_COS") = static_cast<int>(obs::PathCos);
    obs_module.attr("PATH_SIN") = static_cast<int>(obs::PathSin);
    obs_module.attr("DIM") = static_cast<int>(obs::Dim);

    py::module_ gt_module = m.def_submodule("gt", "Индексы вектора истинного состояния state (float64).");
    gt_module.attr("X") = static_cast<int>(gt::X);
    gt_module.attr("Y") = static_cast<int>(gt::Y);
    gt_module.attr("THETA") = static_cast<int>(gt::Theta);
    gt_module.attr("VX") = static_cast<int>(gt::Vx);
    gt_module.attr("VY") = static_cast<int>(gt::Vy);
    gt_module.attr("YAW_RATE") = static_cast<int>(gt::YawRate);
    gt_module.attr("WHEEL_SPEED_LEFT") = static_cast<int>(gt::WheelSpeedLeft);
    gt_module.attr("WHEEL_SPEED_RIGHT") = static_cast<int>(gt::WheelSpeedRight);
    gt_module.attr("GOAL_X") = static_cast<int>(gt::GoalX);
    gt_module.attr("GOAL_Y") = static_cast<int>(gt::GoalY);
    gt_module.attr("CONTACT_FORCE") = static_cast<int>(gt::ContactForce);
    gt_module.attr("PATH_DISTANCE") = static_cast<int>(gt::PathDistance);
    gt_module.attr("DIM") = static_cast<int>(gt::Dim);
}

void bind_world(py::module_& m) {
    py::class_<World, std::shared_ptr<World>>(m, "World", "Статический мир: сетка занятости и аналитические препятствия.")
        .def(py::init([](int width, int height, double cell_size, const Pair& origin) {
                 return std::make_shared<World>(width, height, cell_size, Vec2{origin[0], origin[1]});
             }),
             "width"_a, "height"_a, "cell_size"_a, "origin"_a = Pair{0.0, 0.0},
             "Пустая сетка width × height клеток. После правок вызовите finalize().")
        .def("set_cell", &World::set_cell, "ix"_a, "iy"_a, "occupied"_a = true)
        .def("fill_box",
             [](World& w, double xmin, double ymin, double xmax, double ymax, bool occupied) {
                 w.fill_box({{xmin, ymin}, {xmax, ymax}}, occupied);
             },
             "xmin"_a, "ymin"_a, "xmax"_a, "ymax"_a, "occupied"_a = true,
             "Растеризует прямоугольник в сетку.")
        .def("add_box",
             [](World& w, double xmin, double ymin, double xmax, double ymax) { w.add_box({{xmin, ymin}, {xmax, ymax}}); },
             "xmin"_a, "ymin"_a, "xmax"_a, "ymax"_a, "Аналитический прямоугольник (точная геометрия).")
        .def("add_circle", [](World& w, double x, double y, double radius) { w.add_circle({{x, y}, radius}); },
             "x"_a, "y"_a, "radius"_a, "Аналитический круг (колонна).")
        .def("finalize", &World::finalize, "Строит ускоряющую структуру. Обязателен после правок.")
        .def_property_readonly("finalized", &World::finalized)
        .def_property_readonly("width", &World::width, "ширина сетки [клеток]")
        .def_property_readonly("height", &World::height, "высота сетки [клеток]")
        .def_property_readonly("cell_size", &World::cell_size, "сторона клетки [м]")
        .def_property_readonly("origin", [](const World& w) { return Pair{w.origin().x, w.origin().y}; })
        .def_property_readonly(
            "bounds",
            [](const World& w) {
                const Aabb b = w.bounds();
                return std::array<double, 4>{b.min.x, b.min.y, b.max.x, b.max.y};
            },
            "границы мира: (xmin, ymin, xmax, ymax) [м]")
        .def_property_readonly(
            "grid",
            [](py::object self) {
                const World& w = self.cast<const World&>();
                return view(w.cells().data(), {w.height(), w.width()}, self);
            },
            "сетка занятости uint8 [height, width] без копирования; grid[iy, ix], строка 0 — нижний край мира")
        .def_property_readonly("boxes",
                               [](const World& w) {
                                   std::vector<std::array<double, 4>> out;
                                   for (const Aabb& b : w.boxes()) out.push_back({b.min.x, b.min.y, b.max.x, b.max.y});
                                   return out;
                               })
        .def_property_readonly("circles",
                               [](const World& w) {
                                   std::vector<Triple> out;
                                   for (const Circle& c : w.circles()) out.push_back({c.center.x, c.center.y, c.radius});
                                   return out;
                               })
        .def("occupied", &World::occupied, "ix"_a, "iy"_a, "Клетка занята? За краем сетки — True.")
        .def("is_free",
             [](World& w, double x, double y, double radius) {
                 if (!w.finalized()) w.finalize();
                 return w.is_free({x, y}, radius);
             },
             "x"_a, "y"_a, "radius"_a, "Круг не задевает препятствий и лежит внутри мира.")
        .def("raycast",
             [](World& w, double x, double y, double angle, double max_range) {
                 if (!w.finalized()) w.finalize();
                 return w.raycast({x, y}, {std::cos(angle), std::sin(angle)}, max_range);
             },
             "x"_a, "y"_a, "angle"_a, "max_range"_a, "Дальность до первого препятствия вдоль луча [м].")
        .def_static("arena", [](double width, double height, double cell_size) { return World::arena(width, height, cell_size); },
                    "width"_a, "height"_a, "cell_size"_a = 0.05, "Пустая арена со стенами по периметру.")
        .def_static(
            "maze",
            [](int cols, int rows, double corridor, double wall, double cell_size, std::uint64_t seed, double loop_prob) {
                return World::maze(cols, rows, corridor, wall, cell_size, seed, loop_prob);
            },
            "cols"_a = 5, "rows"_a = 5, "corridor"_a = 1.2, "wall"_a = 0.2, "cell_size"_a = 0.05, "seed"_a = 0,
            "loop_prob"_a = 0.0, "Случайный лабиринт; loop_prob > 0 добавляет петли.")
        .def_static(
            "cluttered",
            [](double width, double height, double cell_size, int num_boxes, int num_pillars, std::uint64_t seed) {
                return World::cluttered(width, height, cell_size, num_boxes, num_pillars, seed);
            },
            "width"_a, "height"_a, "cell_size"_a = 0.05, "num_boxes"_a = 6, "num_pillars"_a = 6, "seed"_a = 0,
            "Арена со случайными прямоугольниками и колоннами.")
        .def_static(
            "from_ascii",
            [](const std::vector<std::string>& rows, double cell_size) {
                const std::vector<std::string_view> views(rows.begin(), rows.end());
                return World::from_ascii(views, cell_size);
            },
            "rows"_a, "cell_size"_a, "Карта из строк: '#' — стена. Первая строка — верх карты.")
        .def_static(
            "from_grid",
            [](const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& grid, double cell_size,
               const Pair& origin) {
                if (grid.ndim() != 2) throw py::value_error("grid must be a 2-D array [height, width]");
                auto world = std::make_shared<World>(static_cast<int>(grid.shape(1)), static_cast<int>(grid.shape(0)),
                                                     cell_size, Vec2{origin[0], origin[1]});
                const auto cells = grid.unchecked<2>();
                for (py::ssize_t iy = 0; iy < grid.shape(0); ++iy) {
                    for (py::ssize_t ix = 0; ix < grid.shape(1); ++ix) {
                        if (cells(iy, ix) != 0) world->set_cell(static_cast<int>(ix), static_cast<int>(iy), true);
                    }
                }
                world->finalize();
                return world;
            },
            "grid"_a, "cell_size"_a, "origin"_a = Pair{0.0, 0.0},
            "Мир из массива занятости [height, width]: ненулевой элемент — стена; строка 0 — нижний край.");
}

void bind_simulator(py::module_& m) {
    py::class_<Simulator>(m, "Simulator", "Один симулятор: физика, сенсоры и задача навигации.")
        .def(py::init([](const SimConfig& config, const std::shared_ptr<World>& world) {
                 return std::make_unique<Simulator>(config, finalized(world));
             }),
             "config"_a, "world"_a)
        .def("reset",
             [](Simulator& sim, std::uint64_t seed, const std::optional<Triple>& start, const std::optional<Pair>& goal) {
                 reset_any(sim, seed, start, goal);
             },
             "seed"_a = 0, "start"_a = py::none(), "goal"_a = py::none(),
             "Новый эпизод. Без start/goal — случайные старт и цель, определяемые seed;\n"
             "иначе start=(x, y, theta), goal=(x, y), а seed задаёт шум сенсоров.")
        .def("step", [](Simulator& sim, double left, double right) { return sim.step({left, right}); },
             "left"_a, "right"_a, py::call_guard<py::gil_scoped_release>(),
             "Шаг управления. left/right — уставки угловых скоростей колёс [рад/с].")
        .def("set_world",
             [](Simulator& sim, const std::shared_ptr<World>& world) { sim.set_world(finalized(world)); }, "world"_a,
             "Заменяет мир; после этого обязателен reset().")
        // ---- наблюдения без копирования ----
        .def_property_readonly(
            "lidar", [](py::object self) { return view(self.cast<const Simulator&>().lidar(), self); },
            "дальности лидара float32 [num_rays], вид без копирования; меняется при reset()/step()")
        .def_property_readonly(
            "proprio", [](py::object self) { return view(self.cast<const Simulator&>().proprio(), self); },
            "IMU, одометрия и цель: float32 [obs.DIM], вид без копирования (индексы — roversim.obs)")
        .def_property_readonly(
            "state", [](py::object self) { return view(self.cast<const Simulator&>().state(), self); },
            "истинное состояние float64 [gt.DIM], вид без копирования (индексы — roversim.gt)")
        .def_property_readonly("last_result", [](const Simulator& sim) { return sim.last_result(); })
        // ---- состояние эпизода ----
        .def_property_readonly("time", &Simulator::time, "время эпизода [с]")
        .def_property_readonly("step_count", &Simulator::step_count)
        .def_property_readonly("episode", &Simulator::episode, "порядковый номер эпизода")
        .def_property_readonly("pose",
                               [](const Simulator& sim) {
                                   const Pose2& p = sim.robot_state().pose;
                                   return Triple{p.x, p.y, p.theta};
                               },
                               "истинная поза (x, y, theta)")
        .def_property_readonly("odom_pose",
                               [](const Simulator& sim) {
                                   const Pose2& p = sim.odom_pose();
                                   return Triple{p.x, p.y, p.theta};
                               },
                               "поза по одометрии (x, y, theta)")
        .def_property_readonly("goal", [](const Simulator& sim) { return Pair{sim.goal().x, sim.goal().y}; })
        .def_property_readonly(
            "obstacles",
            [](const Simulator& sim) {
                const std::span<const DynamicObstacle> obstacles = sim.obstacles();
                py::array_t<double> out({static_cast<py::ssize_t>(obstacles.size()), py::ssize_t{5}});
                auto rows = out.mutable_unchecked<2>();
                for (std::size_t i = 0; i < obstacles.size(); ++i) {
                    const auto r = static_cast<py::ssize_t>(i);
                    rows(r, 0) = obstacles[i].position.x;
                    rows(r, 1) = obstacles[i].position.y;
                    rows(r, 2) = obstacles[i].velocity.x;
                    rows(r, 3) = obstacles[i].velocity.y;
                    rows(r, 4) = obstacles[i].radius;
                }
                return out;
            },
            "подвижные препятствия: массив [N, 5] — x, y, vx, vy, радиус (истинные значения)")
        .def_property_readonly("shortest_path_length", &Simulator::shortest_path_length,
                               "длина кратчайшего пути от старта до цели [м] — для метрики SPL")
        .def("shortest_path",
             [](const Simulator& sim) {
                 const std::vector<Vec2> path = sim.shortest_path();
                 py::array_t<double> out({static_cast<py::ssize_t>(path.size()), py::ssize_t{2}});
                 auto points = out.mutable_unchecked<2>();
                 for (std::size_t i = 0; i < path.size(); ++i) {
                     points(static_cast<py::ssize_t>(i), 0) = path[i].x;
                     points(static_cast<py::ssize_t>(i), 1) = path[i].y;
                 }
                 return out;
             },
             "Кратчайший путь от текущего положения до цели: массив [N, 2].")
        // ---- модель ----
        .def_property_readonly("config", [](const Simulator& sim) { return sim.config(); }, "копия конфигурации")
        .def_property_readonly("world", [](const Simulator& sim) { return std::const_pointer_cast<World>(sim.world_ptr()); })
        .def_property_readonly("num_rays", [](const Simulator& sim) { return sim.lidar().size(); })
        .def_property_readonly("control_dt", [](const Simulator& sim) { return sim.config().control_dt(); })
        .def_property_readonly("lidar_angles",
                               [](const Simulator& sim) {
                                   const Lidar2D& l = sim.lidar_model();
                                   return lidar_angles(l.angle_min(), l.angle_increment(), sim.lidar().size());
                               },
                               "углы лучей в СК сенсора [рад], массив [num_rays]")
        .def_property_readonly("lidar_range_max", [](const Simulator& sim) { return sim.config().lidar.range_max; })
        .def_property_readonly("lidar_mount",
                               [](const Simulator& sim) {
                                   const Pose2& mount = sim.config().lidar.mount;
                                   return Triple{mount.x, mount.y, mount.theta};
                               },
                               "поза лидара в СК робота: (x, y, theta)")
        .def_property_readonly("robot_radius", [](const Simulator& sim) { return sim.config().robot.radius; })
        .def_property_readonly("wheel_radius", [](const Simulator& sim) { return sim.config().robot.wheel_radius; })
        .def_property_readonly("track_width", [](const Simulator& sim) { return sim.config().robot.track_width; })
        .def_property_readonly("max_wheel_speed", [](const Simulator& sim) { return sim.config().robot.max_wheel_speed; });

    py::class_<VecSimulator>(m, "VecSimulator", "Пакет из N независимых сред с общими буферами наблюдений.")
        .def(py::init([](const SimConfig& config, const py::object& worlds, int num_envs, int num_threads, bool autoreset) {
                 return std::make_unique<VecSimulator>(config, world_list(worlds), num_envs, num_threads, autoreset);
             }),
             "config"_a, "worlds"_a, "num_envs"_a, "num_threads"_a = 0, "autoreset"_a = true,
             "worlds — один World или список (среда i работает в worlds[i % len]).\n"
             "num_threads = 0 — по числу ядер.")
        .def("reset", &VecSimulator::reset, "seed"_a = 0, py::call_guard<py::gil_scoped_release>(),
             "Перезапускает все среды.")
        .def("step",
             [](VecSimulator& vec, const py::array_t<double, py::array::c_style | py::array::forcecast>& actions) {
                 if (actions.size() != static_cast<py::ssize_t>(vec.num_envs()) * 2) {
                     throw py::value_error("actions must have shape [num_envs, 2]");
                 }
                 const std::span<const double> span(actions.data(), static_cast<std::size_t>(actions.size()));
                 py::gil_scoped_release release;
                 vec.step(span);
             },
             "actions"_a, "Шаг всех сред. actions — [num_envs, 2]: уставки колёс (левое, правое) [рад/с].")
        .def_property_readonly("num_envs", &VecSimulator::num_envs)
        .def_property_readonly("num_rays", &VecSimulator::num_rays)
        .def_property_readonly("num_threads", &VecSimulator::num_threads)
        .def_property_readonly(
            "lidar",
            [](py::object self) {
                const auto& v = self.cast<const VecSimulator&>();
                return view(v.lidar().data(), {v.num_envs(), v.num_rays()}, self);
            },
            "float32 [num_envs, num_rays], вид без копирования")
        .def_property_readonly(
            "proprio",
            [](py::object self) {
                const auto& v = self.cast<const VecSimulator&>();
                return view(v.proprio().data(), {v.num_envs(), static_cast<py::ssize_t>(obs::Dim)}, self);
            },
            "float32 [num_envs, obs.DIM], вид без копирования")
        .def_property_readonly(
            "state",
            [](py::object self) {
                const auto& v = self.cast<const VecSimulator&>();
                return view(v.state().data(), {v.num_envs(), static_cast<py::ssize_t>(gt::Dim)}, self);
            },
            "float64 [num_envs, gt.DIM], вид без копирования")
        .def_property_readonly("rewards", [](py::object self) { return view(self.cast<const VecSimulator&>().rewards(), self); },
                               "float32 [num_envs]")
        .def_property_readonly("terminated", [](py::object self) { return flag_view(self.cast<const VecSimulator&>().terminated(), self); })
        .def_property_readonly("truncated", [](py::object self) { return flag_view(self.cast<const VecSimulator&>().truncated(), self); })
        .def_property_readonly("collision", [](py::object self) { return flag_view(self.cast<const VecSimulator&>().collision(), self); })
        .def_property_readonly("success", [](py::object self) { return flag_view(self.cast<const VecSimulator&>().success(), self); })
        .def("env", [](VecSimulator& vec, int index) -> Simulator& { return vec.env(index); }, "index"_a,
             py::return_value_policy::reference_internal, "Отдельная среда пакета (для отладки и отрисовки).");
}

void bind_shm(py::module_& m) {
    py::class_<ShmServer>(m, "ShmServer", "Сервер симулятора в разделяемой памяти.")
        .def(py::init([](const std::string& name, const SimConfig& config, const std::shared_ptr<World>& world) {
                 return std::make_unique<ShmServer>(name, config, finalized(world));
             }),
             "name"_a, "config"_a, "world"_a)
        .def_property_readonly("name", &ShmServer::name)
        .def_property_readonly("simulator", [](ShmServer& s) -> Simulator& { return s.simulator(); },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("last_command_seconds", &ShmServer::last_command_seconds,
                               "Время выполнения последней команды сервером [с].")
        .def_property("spin_time", &ShmServer::spin_time, &ShmServer::set_spin_time,
                      "Время активного ожидания команды перед сном на futex [с] (по умолчанию 200 мкс).")
        .def("serve_once", [](ShmServer& s, int timeout_ms) { return static_cast<int>(s.serve_once(timeout_ms)); },
             "timeout_ms"_a = 100, py::call_guard<py::gil_scoped_release>(),
             "Ждёт и выполняет одну команду. Возвращает её код: 0 — не было, 1 — reset, 2 — step, 3 — close.");

    py::class_<ShmClient>(m, "ShmClient", "Клиент симулятора, работающего в другом процессе.")
        .def(py::init<const std::string&, double, bool>(), "name"_a = "/roversim", "timeout"_a = 5.0, "read_only"_a = false,
             "Подключается к сегменту. read_only — наблюдатель: только чтение и snapshot().")
        .def("reset",
             [](ShmClient& c, std::uint64_t seed, const std::optional<Triple>& start, const std::optional<Pair>& goal) {
                 reset_any(c, seed, start, goal);
             },
             "seed"_a = 0, "start"_a = py::none(), "goal"_a = py::none())
        .def("step", &ShmClient::step, "left"_a, "right"_a, py::call_guard<py::gil_scoped_release>(),
             "Шаг управления: уставки колёс [рад/с]. Блокируется до ответа сервера.")
        .def("close_server", &ShmClient::close_server, py::call_guard<py::gil_scoped_release>(),
             "Просит сервер завершить работу.")
        .def_property("spin_time", &ShmClient::spin_time, &ShmClient::set_spin_time,
                      "Время активного ожидания ответа перед сном на futex [с] (по умолчанию 200 мкс).")
        .def_property_readonly("lidar", [](py::object self) { return view(self.cast<const ShmClient&>().lidar(), self); },
                               "float32 [num_rays] — вид прямо в разделяемую память")
        .def_property_readonly("proprio", [](py::object self) { return view(self.cast<const ShmClient&>().proprio(), self); })
        .def_property_readonly("state", [](py::object self) { return view(self.cast<const ShmClient&>().state(), self); })
        .def("snapshot",
             [](const ShmClient& c, double timeout) -> py::object {
                 py::array_t<float> lidar(static_cast<py::ssize_t>(c.lidar().size()));
                 py::array_t<float> proprio(static_cast<py::ssize_t>(c.proprio().size()));
                 py::array_t<double> state(static_cast<py::ssize_t>(c.state().size()));
                 const bool ok = c.snapshot({lidar.mutable_data(), c.lidar().size()}, {proprio.mutable_data(), c.proprio().size()},
                                            {state.mutable_data(), c.state().size()}, timeout);
                 if (!ok) return py::none();
                 return py::make_tuple(lidar, proprio, state);
             },
             "timeout"_a = 0.1,
             "Согласованная копия (lidar, proprio, state); None, если сервер всё время занят записью.")
        .def_property_readonly("time", &ShmClient::time)
        .def_property_readonly("step_count", &ShmClient::step_count)
        .def_property_readonly("episode", &ShmClient::episode)
        .def_property_readonly("num_rays", [](const ShmClient& c) { return c.header().num_rays; })
        .def_property_readonly("control_dt", [](const ShmClient& c) { return c.header().control_dt; })
        .def_property_readonly("lidar_angles",
                               [](const ShmClient& c) {
                                   const ShmHeader& h = c.header();
                                   return lidar_angles(h.lidar_angle_min, h.lidar_angle_increment, h.num_rays);
                               })
        .def_property_readonly("lidar_range_max", [](const ShmClient& c) { return c.header().lidar_range_max; })
        .def_property_readonly("lidar_mount",
                               [](const ShmClient& c) {
                                   const ShmHeader& h = c.header();
                                   return Triple{h.lidar_mount[0], h.lidar_mount[1], h.lidar_mount[2]};
                               })
        .def_property_readonly("robot_radius", [](const ShmClient& c) { return c.header().robot_radius; })
        .def_property_readonly("wheel_radius", [](const ShmClient& c) { return c.header().wheel_radius; })
        .def_property_readonly("track_width", [](const ShmClient& c) { return c.header().track_width; })
        .def_property_readonly("max_wheel_speed", [](const ShmClient& c) { return c.header().max_wheel_speed; });
}

void bind_viz(py::module_& m) {
    py::class_<viz::Renderer>(m, "Renderer", "Программный растеризатор вида сверху.")
        .def(py::init<int, int>(), "width"_a = 512, "height"_a = 512)
        .def("render",
             [](viz::Renderer& renderer, const Simulator& sim) {
                 const viz::Image& image = renderer.render(sim);
                 // Кадр переиспользуется рендерером, поэтому наружу отдаётся копия.
                 return py::array_t<std::uint8_t>({image.height, image.width, 3}, image.rgb.data());
             },
             "sim"_a, "Кадр uint8 [height, width, 3] (RGB).")
        .def("render_ansi", [](viz::Renderer& renderer, const Simulator& sim) { return viz::to_ansi(renderer.render(sim)); },
             "sim"_a, "Кадр для терминала: текст с ANSI-цветами.");

    py::class_<viz::WebViewer>(m, "WebViewer", "Живой просмотр в браузере (HTTP + WebSocket).")
        .def(py::init<std::uint16_t, bool, double>(), "port"_a = 8765, "bind_all"_a = false, "max_fps"_a = 30.0,
             "port = 0 — любой свободный. bind_all — слушать все интерфейсы, а не только 127.0.0.1.")
        .def_property_readonly("port", &viz::WebViewer::port)
        .def_property_readonly("num_clients", &viz::WebViewer::num_clients)
        .def("publish", &viz::WebViewer::publish, "sim"_a, "force"_a = false,
             "Публикует состояние; можно вызывать после каждого шага.")
        .def("teleop",
             [](const viz::WebViewer& viewer) -> py::object {
                 double v = 0.0, w = 0.0;
                 if (!viewer.teleop(v, w)) return py::none();
                 return py::make_tuple(v, w);
             },
             "Команда с клавиатуры из браузера: (v, w) в долях максимума или None.");
}


// ------------------------------------------------------------------ навигация

py::array_t<double> points_array(std::span<const Vec2> points) {
    py::array_t<double> out({static_cast<py::ssize_t>(points.size()), py::ssize_t{2}});
    auto rows = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < points.size(); ++i) {
        rows(static_cast<py::ssize_t>(i), 0) = points[i].x;
        rows(static_cast<py::ssize_t>(i), 1) = points[i].y;
    }
    return out;
}

std::span<const float> scan_span(const py::array_t<float, py::array::c_style | py::array::forcecast>& ranges) {
    return {ranges.data(), static_cast<std::size_t>(ranges.size())};
}

void bind_nav(py::module_& m) {
    py::module_ nav = m.def_submodule("nav", "Автономная навигация: SLAM, глобальный и локальный планировщики.");
    using namespace roversim::nav;

    py::class_<ScanGeometry>(nav, "ScanGeometry", "Геометрия скана лидара.")
        .def(py::init<>())
        .def_readwrite("num_rays", &ScanGeometry::num_rays)
        .def_readwrite("angle_min", &ScanGeometry::angle_min, "угол первого луча в СК сенсора [рад]")
        .def_readwrite("angle_increment", &ScanGeometry::angle_increment, "шаг между лучами [рад]")
        .def_readwrite("range_max", &ScanGeometry::range_max, "дальность «нет отражения» [м]")
        .def_property(
            "mount", [](const ScanGeometry& g) { return Triple{g.mount.x, g.mount.y, g.mount.theta}; },
            [](ScanGeometry& g, const Triple& v) { g.mount = {v[0], v[1], v[2]}; }, "поза сенсора в СК робота");

    py::class_<MapConfig>(nav, "MapConfig", "Параметры карты занятости.")
        .def(py::init<>())
        .def_readwrite("resolution", &MapConfig::resolution, "сторона клетки [м]")
        .def_readwrite("size_x", &MapConfig::size_x, "размер карты по X [м]")
        .def_readwrite("size_y", &MapConfig::size_y, "размер карты по Y [м]")
        .def_readwrite("log_odds_hit", &MapConfig::log_odds_hit)
        .def_readwrite("log_odds_miss", &MapConfig::log_odds_miss)
        .def_readwrite("log_odds_min", &MapConfig::log_odds_min)
        .def_readwrite("log_odds_max", &MapConfig::log_odds_max)
        .def_readwrite("occupied_threshold", &MapConfig::occupied_threshold)
        .def_readwrite("free_threshold", &MapConfig::free_threshold)
        .def_readwrite("free_range", &MapConfig::free_range)
        .def_readwrite("surface_clearance", &MapConfig::surface_clearance)
        .def_readwrite("floor_confirmations", &MapConfig::floor_confirmations);

    py::class_<SlamConfig>(nav, "SlamConfig", "Параметры сеточного SLAM.")
        .def(py::init<>())
        .def_readwrite("map", &SlamConfig::map)
        .def_readwrite("scan_matching", &SlamConfig::scan_matching, "False — карта по одной одометрии")
        .def_readwrite("warmup_scans", &SlamConfig::warmup_scans)
        .def_readwrite("match_points", &SlamConfig::match_points)
        .def_readwrite("min_score", &SlamConfig::min_score)
        .def_readwrite("linear_step", &SlamConfig::linear_step)
        .def_readwrite("angular_step", &SlamConfig::angular_step)
        .def_readwrite("refinements", &SlamConfig::refinements)
        .def_readwrite("max_iterations", &SlamConfig::max_iterations)
        .def_readwrite("prior_linear", &SlamConfig::prior_linear)
        .def_readwrite("prior_angular", &SlamConfig::prior_angular);

    py::enum_<PlannerConfig::Algorithm>(nav, "Algorithm", "Алгоритм глобального планировщика.")
        .value("ASTAR", PlannerConfig::Algorithm::AStar)
        .value("THETA_STAR", PlannerConfig::Algorithm::ThetaStar);

    py::class_<PlannerConfig>(nav, "PlannerConfig", "Параметры глобального планировщика.")
        .def(py::init<>())
        .def_readwrite("algorithm", &PlannerConfig::algorithm)
        .def_readwrite("robot_radius", &PlannerConfig::robot_radius)
        .def_readwrite("safety_margin", &PlannerConfig::safety_margin)
        .def_readwrite("soft_radius", &PlannerConfig::soft_radius)
        .def_readwrite("soft_weight", &PlannerConfig::soft_weight)
        .def_readwrite("unknown_penalty", &PlannerConfig::unknown_penalty)
        .def_readwrite("allow_unknown", &PlannerConfig::allow_unknown)
        .def_readwrite("smooth", &PlannerConfig::smooth)
        .def_readwrite("max_expansions", &PlannerConfig::max_expansions);

    py::class_<DwaConfig>(nav, "DwaConfig", "Параметры локального планировщика DWA.")
        .def(py::init<>())
        .def_readwrite("max_speed", &DwaConfig::max_speed)
        .def_readwrite("min_speed", &DwaConfig::min_speed)
        .def_readwrite("max_yaw_rate", &DwaConfig::max_yaw_rate)
        .def_readwrite("max_accel", &DwaConfig::max_accel)
        .def_readwrite("max_yaw_accel", &DwaConfig::max_yaw_accel)
        .def_readwrite("robot_radius", &DwaConfig::robot_radius)
        .def_readwrite("window_time", &DwaConfig::window_time)
        .def_readwrite("horizon", &DwaConfig::horizon)
        .def_readwrite("time_step", &DwaConfig::time_step)
        .def_readwrite("speed_samples", &DwaConfig::speed_samples)
        .def_readwrite("yaw_samples", &DwaConfig::yaw_samples)
        .def_readwrite("safety_margin", &DwaConfig::safety_margin)
        .def_readwrite("moving_margin", &DwaConfig::moving_margin)
        .def_readwrite("escape_speed", &DwaConfig::escape_speed)
        .def_readwrite("weight_goal", &DwaConfig::weight_goal)
        .def_readwrite("weight_heading", &DwaConfig::weight_heading)
        .def_readwrite("weight_clearance", &DwaConfig::weight_clearance)
        .def_readwrite("weight_speed", &DwaConfig::weight_speed)
        .def_readwrite("weight_survival", &DwaConfig::weight_survival)
        .def_readwrite("clearance_cap", &DwaConfig::clearance_cap)
        .def_readwrite("grid_resolution", &DwaConfig::grid_resolution)
        .def_readwrite("grid_half_size", &DwaConfig::grid_half_size);

    py::class_<TrackerConfig>(nav, "TrackerConfig", "Параметры сопровождения подвижных объектов.")
        .def(py::init<>())
        .def_readwrite("range_jump", &TrackerConfig::range_jump)
        .def_readwrite("min_points", &TrackerConfig::min_points)
        .def_readwrite("max_size", &TrackerConfig::max_size)
        .def_readwrite("max_range", &TrackerConfig::max_range)
        .def_readwrite("gate", &TrackerConfig::gate)
        .def_readwrite("alpha", &TrackerConfig::alpha)
        .def_readwrite("beta", &TrackerConfig::beta)
        .def_readwrite("confirm_hits", &TrackerConfig::confirm_hits)
        .def_readwrite("confirm_travel", &TrackerConfig::confirm_travel)
        .def_readwrite("max_misses", &TrackerConfig::max_misses)
        .def_readwrite("max_speed", &TrackerConfig::max_speed)
        .def_readwrite("park_time", &TrackerConfig::park_time)
        .def_readwrite("park_travel", &TrackerConfig::park_travel);

    py::class_<NavigatorConfig>(nav, "NavigatorConfig", "Параметры навигатора.")
        .def(py::init<>())
        .def_readwrite("slam", &NavigatorConfig::slam)
        .def_readwrite("planner", &NavigatorConfig::planner)
        .def_readwrite("dwa", &NavigatorConfig::dwa)
        .def_readwrite("tracker", &NavigatorConfig::tracker)
        .def_readwrite("control_dt", &NavigatorConfig::control_dt)
        .def_readwrite("wheel_radius", &NavigatorConfig::wheel_radius)
        .def_readwrite("track_width", &NavigatorConfig::track_width)
        .def_readwrite("max_wheel_speed", &NavigatorConfig::max_wheel_speed)
        .def_readwrite("replan_period", &NavigatorConfig::replan_period)
        .def_readwrite("replan_min_period", &NavigatorConfig::replan_min_period)
        .def_readwrite("path_check_distance", &NavigatorConfig::path_check_distance)
        .def_readwrite("lookahead", &NavigatorConfig::lookahead)
        .def_readwrite("goal_tolerance", &NavigatorConfig::goal_tolerance)
        .def_readwrite("obstacle_range", &NavigatorConfig::obstacle_range)
        .def_readwrite("rotate_threshold", &NavigatorConfig::rotate_threshold)
        .def_readwrite("track_obstacles", &NavigatorConfig::track_obstacles)
        .def_readwrite("startup_time", &NavigatorConfig::startup_time)
        .def_readwrite("stuck_time", &NavigatorConfig::stuck_time)
        .def_readwrite("backup_time", &NavigatorConfig::backup_time)
        .def_readwrite("backup_speed", &NavigatorConfig::backup_speed);

    nav.def("navigator_config", &navigator_config, "robot"_a, "control_dt"_a,
            "Настройки навигатора, согласованные с параметрами робота.");
    nav.def("scan_geometry", [](const Simulator& sim) { return scan_geometry(sim.lidar_model()); }, "sim"_a,
            "Геометрия скана лидара данного симулятора.");

    py::class_<OccupancyGrid>(nav, "OccupancyGrid", "Карта занятости в лог-шансах.")
        .def(py::init([](const MapConfig& config, const Pair& center) {
                 return std::make_unique<OccupancyGrid>(config, Vec2{center[0], center[1]});
             }),
             "config"_a, "center"_a = Pair{0.0, 0.0})
        .def("integrate_scan",
             [](OccupancyGrid& map, const Triple& sensor_pose, const py::array_t<float, py::array::c_style | py::array::forcecast>& ranges,
                const ScanGeometry& scan) {
                 std::vector<Vec2> dirs(static_cast<std::size_t>(scan.num_rays));
                 for (int i = 0; i < scan.num_rays; ++i) {
                     const double a = scan.angle_min + i * scan.angle_increment;
                     dirs[static_cast<std::size_t>(i)] = {std::cos(a), std::sin(a)};
                 }
                 map.integrate_scan({sensor_pose[0], sensor_pose[1], sensor_pose[2]}, dirs, scan_span(ranges), scan.range_max);
             },
             "sensor_pose"_a, "ranges"_a, "scan"_a, "Встраивает скан, снятый из позы сенсора (x, y, theta) в СК карты.")
        .def_property_readonly("width", &OccupancyGrid::width)
        .def_property_readonly("height", &OccupancyGrid::height)
        .def_property_readonly("resolution", &OccupancyGrid::resolution)
        .def_property_readonly("origin", [](const OccupancyGrid& map) { return Pair{map.origin().x, map.origin().y}; },
                               "координаты левого нижнего угла клетки (0, 0)")
        .def_property_readonly("scans", &OccupancyGrid::scans, "сколько сканов встроено")
        .def_property_readonly(
            "log_odds",
            [](py::object self) {
                const auto& map = self.cast<const OccupancyGrid&>();
                return view(map.data().data(), {map.height(), map.width()}, self);
            },
            "лог-шансы float32 [height, width] без копирования; строка 0 — нижний край карты")
        .def("probability",
             [](const OccupancyGrid& map) {
                 py::array_t<float> out({map.height(), map.width()});
                 auto cells = out.mutable_unchecked<2>();
                 for (int iy = 0; iy < map.height(); ++iy) {
                     for (int ix = 0; ix < map.width(); ++ix) cells(iy, ix) = static_cast<float>(map.probability(ix, iy));
                 }
                 return out;
             },
             "Вероятность занятости каждой клетки: массив [height, width]; 0.5 — неизвестно.")
        .def("states",
             [](const OccupancyGrid& map) {
                 py::array_t<std::uint8_t> out({map.height(), map.width()});
                 auto cells = out.mutable_unchecked<2>();
                 for (int iy = 0; iy < map.height(); ++iy) {
                     for (int ix = 0; ix < map.width(); ++ix) cells(iy, ix) = static_cast<std::uint8_t>(map.state(ix, iy));
                 }
                 return out;
             },
             "Состояния клеток [height, width]: 0 — свободна, 1 — занята, 2 — неизвестна.");

    py::class_<GridSlam>(nav, "GridSlam", "Сеточный SLAM: карта занятости + сопоставление сканов.")
        .def(py::init([](const SlamConfig& config, const ScanGeometry& scan, const Triple& pose) {
                 return std::make_unique<GridSlam>(config, scan, Pose2{pose[0], pose[1], pose[2]});
             }),
             "config"_a, "scan"_a, "initial_pose"_a)
        .def("update",
             [](GridSlam& slam, const Triple& odom, const py::array_t<float, py::array::c_style | py::array::forcecast>& ranges) {
                 const Pose2& p = slam.update({odom[0], odom[1], odom[2]}, scan_span(ranges));
                 return Triple{p.x, p.y, p.theta};
             },
             "odom_pose"_a, "ranges"_a, "Локализация и встраивание скана. Возвращает оценку позы (x, y, theta).")
        .def_property_readonly("pose",
                               [](const GridSlam& slam) {
                                   const Pose2& p = slam.pose();
                                   return Triple{p.x, p.y, p.theta};
                               })
        .def_property_readonly("map", [](GridSlam& slam) -> const OccupancyGrid& { return slam.map(); },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("match_score", &GridSlam::match_score, "оценка последнего сопоставления, [-1, 1]")
        .def_property_readonly("matched", &GridSlam::matched, "принята ли поза от сопоставления сканов");

    py::class_<GridPlanner>(nav, "GridPlanner", "Глобальный планировщик: A* / Theta* на карте стоимости.")
        .def(py::init<const PlannerConfig&>(), "config"_a = PlannerConfig{})
        .def("set_map", [](GridPlanner& planner, const OccupancyGrid& map) { planner.set_map(map); }, "map"_a,
             "Строит карту стоимости по карте занятости.")
        .def("set_map", [](GridPlanner& planner, const std::shared_ptr<World>& world) { planner.set_map(*finalized(world)); },
             "world"_a, "Строит карту стоимости по истинной карте мира.")
        .def("plan",
             [](GridPlanner& planner, const Pair& start, const Pair& goal) -> py::object {
                 std::vector<Vec2> path;
                 if (!planner.plan({start[0], start[1]}, {goal[0], goal[1]}, path)) return py::none();
                 return points_array(path);
             },
             "start"_a, "goal"_a, "Путь — массив [N, 2] — или None, если пути нет.")
        .def("clearance", [](const GridPlanner& planner, double x, double y) { return planner.clearance({x, y}); }, "x"_a, "y"_a,
             "Расстояние от точки до ближайшего препятствия [м].")
        .def_property_readonly("expansions", &GridPlanner::expansions, "клеток раскрыто последним поиском")
        .def_property_readonly("path_cost", &GridPlanner::path_cost)
        .def_property_readonly(
            "cost",
            [](const GridPlanner& planner) {
                py::array_t<float> out({planner.height(), planner.width()});
                std::copy(planner.cost().begin(), planner.cost().end(), out.mutable_data());
                return out;
            },
            "карта стоимости [height, width]: множитель длины пути, 0 — клетка запрещена");

    py::class_<DwaResult>(nav, "DwaResult", "Решение локального планировщика.")
        .def_readonly("v", &DwaResult::v, "линейная скорость [м/с]")
        .def_readonly("w", &DwaResult::w, "угловая скорость [рад/с]")
        .def_readonly("feasible", &DwaResult::feasible)
        .def_readonly("clearance", &DwaResult::clearance)
        .def_readonly("score", &DwaResult::score)
        .def_readonly("admissible", &DwaResult::admissible, "число допустимых траекторий");

    py::class_<DwaPlanner>(nav, "DwaPlanner", "Локальный планировщик: метод динамического окна.")
        .def(py::init<const DwaConfig&>(), "config"_a = DwaConfig{})
        .def("compute",
             [](DwaPlanner& dwa, const py::array_t<double, py::array::c_style | py::array::forcecast>& points, const Pair& target,
                double v, double w, const std::vector<std::array<double, 5>>& moving, bool allow_reverse) {
                 if (points.size() != 0 && (points.ndim() != 2 || points.shape(1) != 2)) {
                     throw py::value_error("points must have shape [N, 2]");
                 }
                 std::vector<Vec2> obstacles(points.size() == 0 ? 0 : static_cast<std::size_t>(points.shape(0)));
                 for (std::size_t i = 0; i < obstacles.size(); ++i) {
                     obstacles[i] = {points.at(static_cast<py::ssize_t>(i), 0), points.at(static_cast<py::ssize_t>(i), 1)};
                 }
                 std::vector<MovingObstacle> movers;
                 for (const auto& o : moving) movers.push_back({{o[0], o[1]}, {o[2], o[3]}, o[4]});
                 return dwa.compute(obstacles, movers, {target[0], target[1]}, v, w, allow_reverse);
             },
             "points"_a, "target"_a, "v"_a, "w"_a, "moving"_a = std::vector<std::array<double, 5>>{}, "allow_reverse"_a = false,
             "Все координаты — в СК робота. points — точки препятствий [N, 2];\n"
             "moving — подвижные препятствия (x, y, vx, vy, радиус); target — локальная цель.")
        .def_property_readonly("trajectory", [](const DwaPlanner& dwa) { return points_array(dwa.trajectory()); },
                               "точки выбранной траектории [N, 2]");

    py::class_<Track>(nav, "Track", "Сопровождаемый объект.")
        .def_readonly("id", &Track::id)
        .def_property_readonly("position", [](const Track& t) { return Pair{t.position.x, t.position.y}; })
        .def_property_readonly("velocity", [](const Track& t) { return Pair{t.velocity.x, t.velocity.y}; })
        .def_readonly("radius", &Track::radius)
        .def_readonly("hits", &Track::hits)
        .def_readonly("moving", &Track::moving);

    py::class_<Navigator>(nav, "Navigator", "Полный контур автономной навигации по лидару и одометрии.")
        .def(py::init([](const NavigatorConfig& config, const ScanGeometry& scan, const Triple& pose) {
                 return std::make_unique<Navigator>(config, scan, Pose2{pose[0], pose[1], pose[2]});
             }),
             "config"_a, "scan"_a, "initial_pose"_a)
        .def("set_goal", [](Navigator& navigator, double x, double y) { navigator.set_goal({x, y}); }, "x"_a, "y"_a,
             "Цель в СК карты.")
        .def("update",
             [](Navigator& navigator, const py::array_t<float, py::array::c_style | py::array::forcecast>& ranges,
                const Triple& odom, double odom_v, double odom_w) {
                 const std::span<const float> scan = scan_span(ranges);
                 const Pose2 pose{odom[0], odom[1], odom[2]};
                 py::gil_scoped_release release;
                 const WheelCommand command = navigator.update(scan, pose, odom_v, odom_w);
                 return Pair{command.left, command.right};
             },
             "ranges"_a, "odom_pose"_a, "odom_v"_a, "odom_w"_a,
             "Один такт управления. Возвращает уставки колёс (левое, правое) [рад/с].")
        .def_property_readonly("status", [](const Navigator& n) { return static_cast<int>(n.status()); },
                               "0 — нет цели, 1 — едет, 2 — цель достигнута, 3 — нет маршрута, 4 — восстановление")
        .def_property_readonly("pose",
                               [](const Navigator& n) {
                                   const Pose2& p = n.pose();
                                   return Triple{p.x, p.y, p.theta};
                               },
                               "оценка позы (x, y, theta) в СК карты")
        .def_property_readonly("goal", [](const Navigator& n) { return Pair{n.goal().x, n.goal().y}; })
        .def_property_readonly("carrot", [](const Navigator& n) { return Pair{n.carrot().x, n.carrot().y}; },
                               "текущая локальная цель")
        .def_property_readonly("path", [](const Navigator& n) { return points_array(n.path()); }, "маршрут [N, 2]")
        .def_property_readonly("tracks", [](const Navigator& n) { return std::vector<Track>(n.tracks().begin(), n.tracks().end()); },
                               "сопровождаемые объекты")
        .def_property_readonly("map", [](Navigator& n) -> const OccupancyGrid& { return n.map(); },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("slam", [](Navigator& n) -> const GridSlam& { return n.slam(); },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("last_command", [](const Navigator& n) { return n.last_command(); })
        .def_property_readonly("trajectory", [](const Navigator& n) { return points_array(n.dwa().trajectory()); },
                               "траектория, выбранная DWA, в СК робота [N, 2]")
        .def_property_readonly("replans", &Navigator::replans, "сколько раз перестраивался маршрут")
        .def_property_readonly("config", [](const Navigator& n) { return n.config(); });
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "roversim: лёгкий детерминированный симулятор ровера с дифференциальным приводом";
    m.attr("__version__") = "0.1.0";
    bind_config(m);
    bind_world(m);
    bind_simulator(m);
    bind_shm(m);
    bind_viz(m);
    bind_nav(m);
}
