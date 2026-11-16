// Тесты ядра. Внешних фреймворков нет: тест — это функция, зарегистрированная
// макросом TEST; запуск `roversim_tests [подстрока имени]`.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "autopilot.hpp"
#include "roversim/nav/navigator.hpp"
#include "roversim/shm.hpp"
#include "roversim/vec_simulator.hpp"
#include "roversim/viz.hpp"

namespace {

using namespace roversim;

// ------------------------------------------------------------ мини-фреймворк

struct TestCase {
    const char* name;
    void (*run)();
};
std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}
struct Registrar {
    Registrar(const char* name, void (*run)()) { registry().push_back({name, run}); }
};
int g_failures = 0;

#define TEST(name)                                           \
    static void test_##name();                               \
    static const Registrar registrar_##name(#name, test_##name); \
    static void test_##name()

#define CHECK(condition)                                                          \
    do {                                                                          \
        if (!(condition)) {                                                       \
            ++g_failures;                                                         \
            std::printf("    FAILED %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        }                                                                         \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                  \
    do {                                                                                         \
        const double check_a = (actual), check_e = (expected), check_t = (tolerance);            \
        if (!(std::abs(check_a - check_e) <= check_t)) {                                         \
            ++g_failures;                                                                        \
            std::printf("    FAILED %s:%d: %s = %.12g, expected %.12g +- %g\n", __FILE__, __LINE__, \
                        #actual, check_a, check_e, check_t);                                     \
        }                                                                                        \
    } while (0)

// ------------------------------------------------------------------ помощники

/// Конфигурация без шумов и без обрыва эпизодов — для проверки физики.
SimConfig quiet_config() {
    SimConfig config;
    config.task.max_steps = 0;
    config.task.terminate_on_collision = false;
    config.lidar.noise_std = 0.0;
    config.lidar.resolution = 0.0;
    config.imu = ImuConfig{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    config.odom = OdomConfig{0, 0.0, 0.0, 0.0};
    return config;
}

/// То же, но ещё и без потерь на трение: колёса в установившемся режиме точно держат уставку.
SimConfig lossless_config() {
    SimConfig config = quiet_config();
    config.robot.rolling_resistance = 0.0;
    config.robot.linear_drag = 0.0;
    config.robot.angular_drag = 0.0;
    return config;
}

std::shared_ptr<const World> open_arena() { return std::make_shared<const World>(World::arena(20.0, 20.0, 0.05)); }
std::shared_ptr<const World> test_maze() {
    return std::make_shared<const World>(World::maze(5, 5, 1.2, 0.2, 0.05, 7, 0.15));
}

const Vec2 kFarGoal{2.0, 2.0};

/// Детерминированное «блуждающее» управление.
WheelCommand wander(int step, int env = 0) {
    const int phase = (step / 40 + env * 7) % 5;
    constexpr double kLeft[5] = {14.0, 16.0, 6.0, 16.0, -6.0};
    constexpr double kRight[5] = {14.0, 6.0, 16.0, 16.0, 6.0};
    return {kLeft[phase], kRight[phase]};
}

std::uint64_t fnv1a(std::uint64_t hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 0x100000001B3ULL;
    return hash;
}

std::uint64_t hash_observation(std::uint64_t hash, const Simulator& sim) {
    hash = fnv1a(hash, sim.lidar().data(), sim.lidar().size_bytes());
    hash = fnv1a(hash, sim.proprio().data(), sim.proprio().size_bytes());
    return fnv1a(hash, sim.state().data(), sim.state().size_bytes());
}

template <typename A, typename B>
bool same_bytes(std::span<const A> a, std::span<const B> b) {
    return a.size_bytes() == b.size_bytes() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// ------------------------------------------------------------------------ ГСЧ

TEST(rng_is_reproducible_and_well_distributed) {
    Rng a(42), b(42), c(43);
    bool same = true, differs = false;
    for (int i = 0; i < 1000; ++i) {
        const std::uint64_t x = a.next_u64();
        same = same && x == b.next_u64();
        differs = differs || x != c.next_u64();
    }
    CHECK(same);
    CHECK(differs);
    CHECK(Rng::stream(1, 0).next_u64() != Rng::stream(1, 1).next_u64());
    CHECK(Rng::stream(1, 0).next_u64() != Rng::stream(2, 0).next_u64());

    Rng rng(7);
    const int n = 400000;
    double sum_u = 0.0, sum_n = 0.0, sum_n2 = 0.0, sum_n4 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double u = rng.uniform();
        CHECK(u >= 0.0 && u < 1.0);
        sum_u += u;
        const double z = rng.normal();
        sum_n += z;
        sum_n2 += z * z;
        sum_n4 += z * z * z * z;
    }
    CHECK_NEAR(sum_u / n, 0.5, 0.003);
    CHECK_NEAR(sum_n / n, 0.0, 0.01);
    CHECK_NEAR(sum_n2 / n, 1.0, 0.01);
    CHECK_NEAR(sum_n4 / n, 3.0, 0.08);  // эксцесс нормального распределения

    int seen[7] = {};
    for (int i = 0; i < 7000; ++i) ++seen[rng.below(7)];
    for (const int count : seen) CHECK(count > 800 && count < 1200);
}

// ------------------------------------------------------------------------ мир

// Независимая эталонная трассировка: перебор всех занятых клеток и примитивов.
double reference_raycast(const World& world, Vec2 o, Vec2 d, double max_range) {
    const auto hit_box = [&](Vec2 lo, Vec2 hi) {
        double t_near = -std::numeric_limits<double>::infinity();
        double t_far = std::numeric_limits<double>::infinity();
        for (int axis = 0; axis < 2; ++axis) {
            const double origin = axis == 0 ? o.x : o.y, dir = axis == 0 ? d.x : d.y;
            const double a = axis == 0 ? lo.x : lo.y, b = axis == 0 ? hi.x : hi.y;
            if (dir == 0.0) {
                if (origin < a || origin > b) return std::numeric_limits<double>::infinity();
                continue;
            }
            const double t1 = (a - origin) / dir, t2 = (b - origin) / dir;
            t_near = std::max(t_near, std::min(t1, t2));
            t_far = std::min(t_far, std::max(t1, t2));
        }
        if (t_near > t_far || t_far < 0.0) return std::numeric_limits<double>::infinity();
        return std::max(t_near, 0.0);
    };

    double best = max_range;
    const double cell = world.cell_size();
    for (int iy = 0; iy < world.height(); ++iy) {
        for (int ix = 0; ix < world.width(); ++ix) {
            if (!world.occupied(ix, iy)) continue;
            const Vec2 lo{world.origin().x + ix * cell, world.origin().y + iy * cell};
            best = std::min(best, hit_box(lo, {lo.x + cell, lo.y + cell}));
        }
    }
    // Край мира: расстояние, на котором луч покидает прямоугольник мира.
    const Aabb b = world.bounds();
    double exit = std::numeric_limits<double>::infinity();
    if (d.x > 0.0) exit = std::min(exit, (b.max.x - o.x) / d.x);
    if (d.x < 0.0) exit = std::min(exit, (b.min.x - o.x) / d.x);
    if (d.y > 0.0) exit = std::min(exit, (b.max.y - o.y) / d.y);
    if (d.y < 0.0) exit = std::min(exit, (b.min.y - o.y) / d.y);
    best = std::min(best, exit);
    for (const Aabb& box : world.boxes()) best = std::min(best, hit_box(box.min, box.max));
    for (const Circle& c : world.circles()) {
        const Vec2 m = o - c.center;
        const double half_b = dot(m, d), cc = dot(m, m) - c.radius * c.radius;
        if (cc <= 0.0) return 0.0;
        const double disc = half_b * half_b - cc;
        if (disc >= 0.0 && -half_b - std::sqrt(disc) >= 0.0) best = std::min(best, -half_b - std::sqrt(disc));
    }
    return best;
}

TEST(raycast_matches_brute_force) {
    // Разная плотность препятствий: в разреженных мирах работает «прыжок» по полю
    // расстояний, в плотных — обычные шаги DDA. Результат обязан совпадать с перебором.
    Rng rng(2024);
    double worst = 0.0;
    int checked = 0;
    for (const double density : {0.0, 0.01, 0.05, 0.2}) {
        for (int trial = 0; trial < 4; ++trial) {
            const int w = 40 + static_cast<int>(rng.below(60)), h = 40 + static_cast<int>(rng.below(60));
            World world(w, h, 0.1, {-1.5, 2.5});
            for (int iy = 0; iy < h; ++iy) {
                for (int ix = 0; ix < w; ++ix) {
                    if (rng.uniform() < density) world.set_cell(ix, iy, true);
                }
            }
            const Aabb b = world.bounds();
            for (int i = 0; i < 3; ++i) {
                world.add_circle({{rng.uniform(b.min.x, b.max.x), rng.uniform(b.min.y, b.max.y)}, rng.uniform(0.1, 0.5)});
                const Vec2 lo{rng.uniform(b.min.x, b.max.x), rng.uniform(b.min.y, b.max.y)};
                world.add_box({lo, {lo.x + rng.uniform(0.1, 0.8), lo.y + rng.uniform(0.1, 0.8)}});
            }
            world.finalize();
            for (int ray = 0; ray < 400; ++ray) {
                const Vec2 origin{rng.uniform(b.min.x, b.max.x), rng.uniform(b.min.y, b.max.y)};
                const double angle = rng.uniform(-kPi, kPi), range = rng.uniform(0.3, 15.0);
                const Vec2 dir{std::cos(angle), std::sin(angle)};
                const double got = world.raycast(origin, dir, range);
                const double expected = reference_raycast(world, origin, dir, range);
                worst = std::max(worst, std::abs(got - expected));
                ++checked;
            }
        }
    }
    CHECK(checked == 6400);
    CHECK_NEAR(worst, 0.0, 1e-9);
}

TEST(raycast_scan_equals_single_rays) {
    World world = World::cluttered(8.0, 6.0, 0.05, 5, 5, 3);
    std::vector<Vec2> dirs;
    for (int i = 0; i < 180; ++i) dirs.push_back({std::cos(i * 0.035), std::sin(i * 0.035)});
    std::vector<float> scan(dirs.size());
    const Vec2 origin{4.0, 3.0};
    const double yaw = 0.7;
    world.raycast_scan(origin, std::cos(yaw), std::sin(yaw), dirs, 6.0, scan);
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        const Vec2 dir = rotate(dirs[i], std::cos(yaw), std::sin(yaw));
        CHECK(scan[i] == static_cast<float>(world.raycast(origin, dir, 6.0)));
    }
}

TEST(collision_contacts_have_no_ghost_edges) {
    // Круг движется вдоль ровной стены из клеток: контакт ровно один, нормаль строго
    // перпендикулярна стене. Стыки клеток не должны порождать лишних контактов.
    const World arena = World::arena(4.0, 4.0, 0.1);
    Contact contacts[8];
    for (double y = 1.0; y < 3.0; y += 0.0137) {
        const std::size_t n = arena.collide_circle({0.35, y}, 0.2, 0.1, contacts);
        CHECK(n == 1);
        CHECK_NEAR(contacts[0].normal.x, 1.0, 0.0);
        CHECK_NEAR(contacts[0].normal.y, 0.0, 0.0);
        CHECK_NEAR(contacts[0].gap, 0.05, 1e-12);
    }
    // Вогнутый угол: два контакта — по одному от каждой стены.
    CHECK(arena.collide_circle({0.32, 0.32}, 0.2, 0.05, contacts) == 2);

    // Выпуклый угол отдельного блока: нормаль направлена по диагонали от вершины.
    World block(40, 40, 0.1);
    block.fill_box({{1.0, 1.0}, {2.0, 2.0}});
    block.finalize();
    CHECK(block.collide_circle({2.1, 2.1}, 0.1, 0.1, contacts) == 1);
    CHECK_NEAR(contacts[0].normal.x, std::sqrt(0.5), 1e-9);
    CHECK_NEAR(contacts[0].normal.y, std::sqrt(0.5), 1e-9);
    CHECK_NEAR(contacts[0].gap, std::sqrt(0.02) - 0.1, 1e-9);
    CHECK(block.is_free({2.3, 2.3}, 0.2));
    CHECK(!block.is_free({2.1, 1.5}, 0.2));
    CHECK(!block.is_free({1.5, 1.5}, 0.2));  // центр внутри препятствия
    CHECK(!block.is_free({0.1, 2.0}, 0.2));  // выход за край мира

    // Аналитические препятствия.
    World shapes(40, 40, 0.1);
    shapes.add_circle({{2.0, 2.0}, 0.5});
    shapes.add_box({{3.0, 0.5}, {3.5, 1.0}});
    shapes.finalize();
    CHECK(shapes.collide_circle({2.9, 2.0}, 0.2, 0.3, contacts) == 1);
    CHECK_NEAR(contacts[0].gap, 0.2, 1e-12);
    CHECK_NEAR(contacts[0].normal.x, 1.0, 1e-12);
    CHECK(shapes.collide_circle({3.25, 1.3}, 0.2, 0.2, contacts) == 1);
    CHECK_NEAR(contacts[0].gap, 0.1, 1e-12);
    CHECK_NEAR(contacts[0].normal.y, 1.0, 1e-12);
}

TEST(world_generators) {
    const World maze = World::maze(4, 3, 1.0, 0.2, 0.1, 5);
    CHECK(maze.width() == 4 * 12 + 2 && maze.height() == 3 * 12 + 2);
    // Идеальный лабиринт связен: в каждую ячейку можно попасть.
    const NavGrid nav(maze, 0.25);
    CHECK(nav.free_count() > 0);
    std::vector<float> field;
    std::vector<std::int32_t> next;
    nav.distance_field(maze.cell_center(6, 6), field, next);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            CHECK(std::isfinite(nav.lookup(field, maze.cell_center(2 + c * 12 + 5, 2 + r * 12 + 5))));
        }
    }
    // Генераторы детерминированы по seed.
    const World again = World::maze(4, 3, 1.0, 0.2, 0.1, 5), other = World::maze(4, 3, 1.0, 0.2, 0.1, 6);
    CHECK(same_bytes(maze.cells(), again.cells()));
    CHECK(!same_bytes(maze.cells(), other.cells()));

    const std::string_view rows[] = {"####", "#..#", "#.##", "####"};
    const World ascii = World::from_ascii(rows, 0.5);
    CHECK(ascii.width() == 4 && ascii.height() == 4);
    CHECK(!ascii.occupied(1, 2) && !ascii.occupied(2, 2) && !ascii.occupied(1, 1) && ascii.occupied(2, 1));
    CHECK(ascii.occupied(-1, 0));  // за краем — стена
}

TEST(navgrid_paths_are_valid_and_short) {
    SimConfig config = quiet_config();
    config.lidar.num_rays = 8;
    Simulator sim(config, test_maze());
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        sim.reset(seed);
        const Vec2 start = sim.robot_state().pose.position();
        const double length = sim.shortest_path_length();
        CHECK(std::isfinite(length));  // старт и цель всегда связаны
        CHECK(length >= distance(start, sim.goal()) - 0.1);
        CHECK(distance(start, sim.goal()) >= config.task.min_goal_distance);

        const std::vector<Vec2> path = sim.shortest_path();
        CHECK(path.size() >= 2);
        double walked = 0.0;
        for (std::size_t i = 1; i < path.size(); ++i) {
            walked += distance(path[i - 1], path[i]);
            CHECK(sim.world().is_free(path[i], config.robot.radius));
        }
        CHECK_NEAR(walked, length, 0.25);
        CHECK(distance(path.back(), sim.goal()) == 0.0);
    }
}

// --------------------------------------------------------------------- физика

TEST(kinematics_follows_exact_arc) {
    // Без потерь колёса держат уставку точно, и движение обязано совпасть с
    // кинематикой дифференциального привода: v = r(wl + wr)/2, w = r(wr − wl)/b.
    const SimConfig config = lossless_config();
    Simulator sim(config, open_arena());
    sim.reset(1, Pose2{10.0, 10.0, 0.3}, kFarGoal);
    const WheelCommand command{6.0, 10.0};
    for (int i = 0; i < 100; ++i) sim.step(command);  // выход на установившийся режим

    const double r = config.robot.wheel_radius, b = config.robot.track_width;
    const double v = r * (command.left + command.right) / 2.0, w = r * (command.right - command.left) / b;
    const RobotState before = sim.robot_state();
    CHECK_NEAR(before.velocity.x, v, 1e-12);
    CHECK_NEAR(before.velocity.y, 0.0, 1e-12);
    CHECK_NEAR(before.yaw_rate, w, 1e-12);

    const int steps = 60;
    for (int i = 0; i < steps; ++i) sim.step(command);
    const RobotState after = sim.robot_state();
    const double t = steps * config.control_dt();
    const double theta = before.pose.theta + w * t;
    CHECK_NEAR(after.pose.x, before.pose.x + v / w * (std::sin(theta) - std::sin(before.pose.theta)), 1e-9);
    CHECK_NEAR(after.pose.y, before.pose.y - v / w * (std::cos(theta) - std::cos(before.pose.theta)), 1e-9);
    CHECK_NEAR(after.pose.theta, wrap_angle(theta), 1e-9);
    // Качение без проскальзывания: путь по энкодеру равен пути точки контакта колеса.
    CHECK_NEAR((after.wheel_angle[0] - before.wheel_angle[0]) * r, (v - w * b / 2.0) * t, 1e-9);
    CHECK_NEAR((after.wheel_angle[1] - before.wheel_angle[1]) * r, (v + w * b / 2.0) * t, 1e-9);
}

TEST(acceleration_matches_newton) {
    const SimConfig config = quiet_config();
    const RobotParams& p = config.robot;
    Simulator sim(config, open_arena());
    sim.reset(1, Pose2{3.0, 10.0, 0.0}, kFarGoal);
    sim.step({20.0, 20.0});
    const double v1 = sim.robot_state().velocity.x;
    for (int i = 0; i < 4; ++i) sim.step({20.0, 20.0});
    const double v2 = sim.robot_state().velocity.x;

    // Оба привода в насыщении: (m + 2·Iw/r²)·a = 2·T/r − Crr·m·g − c·v.
    const double effective_mass = p.mass + 2.0 * p.wheel_inertia / (p.wheel_radius * p.wheel_radius);
    const double expected = (2.0 * p.motor_max_torque / p.wheel_radius - p.rolling_resistance * p.mass * p.gravity -
                             p.linear_drag * 0.5 * (v1 + v2)) / effective_mass;
    CHECK_NEAR((v2 - v1) / (4 * config.control_dt()), expected, 0.005);
    // Строго прямо: ни бокового сноса, ни вращения.
    CHECK_NEAR(sim.robot_state().velocity.y, 0.0, 1e-12);
    CHECK_NEAR(sim.robot_state().yaw_rate, 0.0, 1e-12);
    CHECK_NEAR(sim.robot_state().pose.y, 10.0, 1e-12);

    // Предельная скорость: момент привода уравновешивает сопротивление.
    for (int i = 0; i < 100; ++i) sim.step({20.0, 20.0});
    const double v_max = sim.robot_state().velocity.x;
    CHECK(v_max > 0.98 && v_max < 1.0);
    const double wheel_torque = p.motor_gain * (20.0 - sim.robot_state().wheel_speed[0]);
    // Баланс сил выполняется с точностью порядка dt (схема первого порядка по времени).
    CHECK_NEAR(2.0 * wheel_torque / p.wheel_radius, p.rolling_resistance * p.mass * p.gravity + p.linear_drag * v_max, 2e-3);
}

TEST(wheels_slip_when_traction_is_low) {
    // «Лёд»: тяга ограничена трением mu·m·g, колёса буксуют, а одометрия
    // насчитывает путь, которого не было.
    SimConfig config = quiet_config();
    config.robot.mu_long = 0.05;
    const RobotParams& p = config.robot;
    Simulator sim(config, open_arena());
    sim.reset(1, Pose2{3.0, 10.0, 0.0}, kFarGoal);
    for (int i = 0; i < 20; ++i) sim.step({20.0, 20.0});  // 1 секунда
    const RobotState& s = sim.robot_state();
    const double expected_accel = (p.mu_long - p.rolling_resistance) * p.gravity;
    CHECK_NEAR(s.velocity.x, expected_accel * 1.0, 0.02);
    CHECK(s.wheel_speed[0] * p.wheel_radius > 0.9);  // обод колеса намного быстрее корпуса
    const double true_distance = s.pose.x - 3.0, odom_distance = sim.odom_pose().x - 3.0;
    CHECK(odom_distance > 3.0 * true_distance);
}

TEST(wall_stops_robot_without_penetration) {
    const SimConfig config = quiet_config();
    const RobotParams& p = config.robot;
    Simulator sim(config, open_arena());
    sim.reset(1, Pose2{18.0, 10.0, 0.0}, kFarGoal);
    const double limit = 20.0 - 0.05 - p.radius;  // положение центра при касании стены
    bool collided = false;
    for (int i = 0; i < 80; ++i) {
        collided = sim.step({20.0, 20.0}).collision || collided;
        CHECK(sim.robot_state().pose.x <= limit + 1e-9);
    }
    CHECK(collided);
    CHECK(sim.last_result().collision);
    CHECK_NEAR(sim.robot_state().pose.x, limit, 1e-6);
    CHECK_NEAR(sim.robot_state().velocity.x, 0.0, 1e-6);
    // Моторы упёрлись: колёса стоят, стена держит полную тягу 2·T/r.
    CHECK_NEAR(sim.robot_state().wheel_speed[0], 0.0, 1e-6);
    CHECK_NEAR(sim.state()[gt::ContactForce], 2.0 * p.motor_max_torque / p.wheel_radius, 0.01);

    // Задний ход: робот свободно отъезжает, контакт пропадает.
    for (int i = 0; i < 20; ++i) sim.step({-10.0, -10.0});
    CHECK(!sim.last_result().collision);
    CHECK(sim.robot_state().pose.x < limit - 0.2);
}

TEST(driving_along_wall_is_not_disturbed) {
    // Робот едет параллельно стене с зазором 2 мм. Упреждающие контакты и стыки
    // клеток не должны влиять на движение: траектория совпадает с ездой в чистом поле.
    const SimConfig config = quiet_config();
    Simulator near_wall(config, open_arena()), open_field(config, open_arena());
    near_wall.reset(1, Pose2{2.0, 0.05 + config.robot.radius + 0.002, 0.0}, kFarGoal);
    open_field.reset(1, Pose2{2.0, 10.0, 0.0}, kFarGoal);
    for (int i = 0; i < 200; ++i) {
        near_wall.step({16.0, 16.0});
        open_field.step({16.0, 16.0});
        CHECK(!near_wall.last_result().collision);
    }
    CHECK(near_wall.robot_state().pose.x == open_field.robot_state().pose.x);
    CHECK(near_wall.robot_state().velocity.x == open_field.robot_state().velocity.x);
    CHECK(near_wall.robot_state().pose.x > 9.0);
}

TEST(robot_stays_inside_free_space_under_random_driving) {
    // Длинная «пьяная» езда по лабиринту с ударами о стены под разными углами:
    // корпус никогда не оказывается внутри препятствия.
    SimConfig config = quiet_config();
    config.lidar.num_rays = 8;
    Simulator sim(config, test_maze());
    Rng rng(99);
    for (std::uint64_t episode = 0; episode < 6; ++episode) {
        sim.reset(episode);
        WheelCommand command{};
        for (int i = 0; i < 1500; ++i) {
            if (i % 15 == 0) command = {rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0)};
            sim.step(command);
            Contact contacts[8];
            const std::size_t n = sim.world().collide_circle(sim.robot_state().pose.position(), config.robot.radius, 0.0, contacts);
            for (std::size_t k = 0; k < n; ++k) CHECK(contacts[k].gap > -1e-9);
            CHECK(std::isfinite(sim.robot_state().pose.x) && std::isfinite(sim.robot_state().velocity.x));
        }
    }
}

// -------------------------------------------------------------------- сенсоры

TEST(lidar_noise_statistics) {
    const auto arena = std::make_shared<const World>(World::arena(10.0, 10.0, 0.05));
    const Pose2 pose{4.3, 5.2, 0.4};

    SimConfig exact = quiet_config();
    Simulator truth(exact, arena);
    truth.reset(0, pose, kFarGoal);

    SimConfig noisy = exact;
    noisy.lidar.noise_std = 0.02;
    Simulator sim(noisy, arena);
    double sum = 0.0, sum2 = 0.0;
    std::size_t count = 0;
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        sim.reset(seed, pose, kFarGoal);
        for (std::size_t i = 0; i < sim.lidar().size(); ++i) {
            const double error = sim.lidar()[i] - truth.lidar()[i];
            sum += error;
            sum2 += error * error;
            ++count;
        }
    }
    CHECK_NEAR(sum / count, 0.0, 5e-4);
    CHECK_NEAR(std::sqrt(sum2 / count), 0.02, 5e-4);

    // Квантование: все дальности кратны шагу.
    SimConfig quantized = noisy;
    quantized.lidar.resolution = 0.01;
    Simulator q(quantized, arena);
    q.reset(3, pose, kFarGoal);
    for (const float range : q.lidar()) CHECK_NEAR(range / 0.01, std::round(range / 0.01), 1e-3);

    // Потеря отражений: доля лучей с range_max близка к dropout_prob.
    SimConfig lossy = exact;
    lossy.lidar.dropout_prob = 0.3;
    Simulator d(lossy, arena);
    std::size_t dropped = 0, total = 0;
    for (std::uint64_t seed = 0; seed < 30; ++seed) {
        d.reset(seed, pose, kFarGoal);
        for (const float range : d.lidar()) {
            dropped += range >= static_cast<float>(lossy.lidar.range_max) ? 1 : 0;
            ++total;
        }
    }
    CHECK_NEAR(static_cast<double>(dropped) / static_cast<double>(total), 0.3, 0.02);
}

TEST(lidar_geometry) {
    // Робот в центре квадратной комнаты смотрит вдоль +X: дальности до стен известны.
    const auto room = std::make_shared<const World>(World::arena(6.0, 6.0, 0.05));
    SimConfig config = quiet_config();
    config.lidar.num_rays = 4;  // лучи: назад, вправо, вперёд, влево
    Simulator sim(config, room);
    sim.reset(0, Pose2{2.0, 3.0, 0.0}, kFarGoal);
    const Lidar2D& lidar = sim.lidar_model();
    CHECK_NEAR(lidar.angle_min(), -kPi, 1e-12);
    CHECK_NEAR(lidar.angle_increment(), kPi / 2.0, 1e-12);
    CHECK_NEAR(sim.lidar()[0], 2.0 - 0.05, 1e-6);  // назад (−X)
    CHECK_NEAR(sim.lidar()[1], 3.0 - 0.05, 1e-6);  // вправо (−Y)
    CHECK_NEAR(sim.lidar()[2], 4.0 - 0.05, 1e-6);  // вперёд (+X)
    CHECK_NEAR(sim.lidar()[3], 3.0 - 0.05, 1e-6);  // влево (+Y)

    // Неполный обзор: крайние лучи идут точно по границам сектора.
    LidarConfig sector;
    sector.num_rays = 5;
    sector.fov = kPi / 2.0;
    const Lidar2D narrow(sector);
    CHECK_NEAR(narrow.angle_min(), -kPi / 4.0, 1e-12);
    CHECK_NEAR(narrow.angle_min() + 4 * narrow.angle_increment(), kPi / 4.0, 1e-12);
}

TEST(imu_measures_motion) {
    // Установившаяся дуга: гироскоп показывает w, акселерометр — центростремительное v·w.
    const SimConfig config = lossless_config();
    Simulator sim(config, open_arena());
    sim.reset(1, Pose2{10.0, 10.0, 0.0}, kFarGoal);
    for (int i = 0; i < 100; ++i) sim.step({6.0, 10.0});
    const double v = 0.4, w = 0.05 * 4.0 / 0.3;
    CHECK_NEAR(sim.proprio()[obs::GyroZ], w, 1e-6);
    CHECK_NEAR(sim.proprio()[obs::AccelX], 0.0, 1e-6);
    CHECK_NEAR(sim.proprio()[obs::AccelY], v * w, 1e-6);

    // Разгон по прямой: продольное ускорение равно приращению скорости за шаг.
    sim.reset(1, Pose2{3.0, 10.0, 0.0}, kFarGoal);
    sim.step({20.0, 20.0});
    const double v1 = sim.robot_state().velocity.x;
    sim.step({20.0, 20.0});
    CHECK_NEAR(sim.proprio()[obs::AccelX], (sim.robot_state().velocity.x - v1) / config.control_dt(), 1e-5);

    // Шум неподвижного гироскопа: СКО = плотность / sqrt(T).
    SimConfig noisy = quiet_config();
    noisy.imu.gyro_noise_density = 2e-4;
    Simulator still(noisy, open_arena());
    still.reset(5, Pose2{10.0, 10.0, 0.0}, kFarGoal);
    double sum = 0.0, sum2 = 0.0;
    const int n = 4000;
    for (int i = 0; i < n; ++i) {
        still.step({0.0, 0.0});
        sum += still.proprio()[obs::GyroZ];
        sum2 += static_cast<double>(still.proprio()[obs::GyroZ]) * still.proprio()[obs::GyroZ];
    }
    const double mean = sum / n;
    CHECK_NEAR(std::sqrt(sum2 / n - mean * mean), 2e-4 / std::sqrt(noisy.control_dt()), 1e-4);
}

TEST(odometry_tracks_truth_and_drifts_with_noise) {
    // Без шумов счисление почти совпадает с истиной.
    const SimConfig clean = quiet_config();
    Simulator sim(clean, open_arena());
    sim.reset(1, Pose2{6.0, 6.0, 0.2}, kFarGoal);
    for (int i = 0; i < 400; ++i) sim.step(wander(i));
    const RobotState& truth = sim.robot_state();
    CHECK_NEAR(sim.odom_pose().x, truth.pose.x, 2e-3);
    CHECK_NEAR(sim.odom_pose().y, truth.pose.y, 2e-3);
    CHECK_NEAR(wrap_angle(sim.odom_pose().theta - truth.pose.theta), 0.0, 1e-3);
    CHECK_NEAR(sim.proprio()[obs::OdomV], truth.velocity.x, 1e-3);

    // С шумом и ошибкой калибровки появляется умеренный, но ненулевой дрейф.
    SimConfig noisy = SimConfig{};
    noisy.task.max_steps = 0;
    noisy.task.terminate_on_collision = false;
    Simulator drifting(noisy, open_arena());
    drifting.reset(1, Pose2{6.0, 6.0, 0.2}, kFarGoal);
    for (int i = 0; i < 400; ++i) drifting.step(wander(i));
    const double drift = distance(drifting.odom_pose().position(), drifting.robot_state().pose.position());
    CHECK(drift > 1e-3 && drift < 0.5);
    // Энкодер квантован: скорость колеса кратна одному тику за шаг.
    const double tick_rate = kTwoPi / noisy.odom.ticks_per_rev / noisy.control_dt();
    const double ticks = drifting.proprio()[obs::WheelVelLeft] / tick_rate;
    CHECK_NEAR(ticks, std::round(ticks), 1e-3);
}

// ------------------------------------------------------------ симулятор в целом

TEST(simulation_is_deterministic) {
    const auto world = test_maze();
    const auto run = [&](std::uint64_t seed) {
        Simulator sim(SimConfig{}, world);
        sim.reset(seed);
        std::uint64_t hash = hash_observation(0xCBF29CE484222325ULL, sim);
        std::uint64_t episode_seed = seed;
        for (int i = 0; i < 1500; ++i) {
            const StepResult r = sim.step(wander(i));
            hash = hash_observation(hash, sim);
            hash = fnv1a(hash, &r.reward, sizeof(r.reward));
            if (r.terminated || r.truncated) sim.reset(++episode_seed);
        }
        return hash;
    };
    CHECK(run(5) == run(5));
    CHECK(run(5) != run(6));
}

TEST(task_rewards_and_termination) {
    SimConfig config = quiet_config();
    config.task.terminate_on_collision = true;
    config.task.geodesic = false;
    config.task.max_steps = 30;
    Simulator sim(config, open_arena());

    // Прогресс: награда равна сокращению расстояния до цели минус штраф за шаг.
    sim.reset(1, Pose2{5.0, 10.0, 0.0}, Vec2{9.0, 10.0});
    double total = 0.0;
    for (int i = 0; i < 20; ++i) total += sim.step({16.0, 16.0}).reward;
    const double travelled = sim.robot_state().pose.x - 5.0;
    CHECK_NEAR(total, travelled - 20 * config.task.step_penalty, 1e-9);
    CHECK_NEAR(sim.proprio()[obs::GoalDistance], 4.0 - travelled, 1e-5);
    CHECK_NEAR(sim.proprio()[obs::GoalCos], 1.0, 1e-6);

    // С геодезической метрикой на пустой арене результат тот же (кратчайший путь —
    // прямая), а награда за шаг плавная: нет шагов с нулевым прогрессом.
    {
        SimConfig geodesic = config;
        geodesic.task.geodesic = true;
        Simulator smooth(geodesic, open_arena());
        smooth.reset(1, Pose2{5.0, 10.0, 0.0}, Vec2{9.0, 10.0});
        double sum = 0.0;
        for (int i = 0; i < 20; ++i) {
            const double before = smooth.robot_state().pose.x;
            const double reward = smooth.step({16.0, 16.0}).reward + geodesic.task.step_penalty;
            sum += reward;
            CHECK_NEAR(reward, smooth.robot_state().pose.x - before, 1e-5);
        }
        CHECK_NEAR(sum, smooth.robot_state().pose.x - 5.0, 1e-4);
    }

    // Лимит шагов: truncated, но не terminated.
    for (int i = 0; i < 9; ++i) CHECK(!sim.step({0.0, 0.0}).truncated);
    const StepResult last = sim.step({0.0, 0.0});
    CHECK(last.truncated && !last.terminated);

    // Достижение цели.
    sim.reset(1, Pose2{5.0, 10.0, 0.0}, Vec2{5.6, 10.0});
    StepResult r;
    for (int i = 0; i < 30 && !r.terminated; ++i) r = sim.step({16.0, 16.0});
    CHECK(r.success && r.terminated && !r.collision);
    CHECK(r.reward > config.task.success_reward - 1.0);

    // Столкновение завершает эпизод со штрафом.
    sim.reset(1, Pose2{19.5, 10.0, 0.0}, Vec2{5.0, 10.0});
    r = {};
    for (int i = 0; i < 30 && !r.terminated; ++i) r = sim.step({20.0, 20.0});
    CHECK(r.collision && r.terminated && !r.success);
    CHECK(r.reward < -config.task.collision_penalty + 1.0);

    // Старт внутри стены отклоняется.
    bool thrown = false;
    try {
        sim.reset(1, Pose2{0.1, 10.0, 0.0}, Vec2{5.0, 10.0});
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    CHECK(thrown);
}

TEST(autopilot_reaches_goals_in_maze) {
    // Сквозная проверка: по кратчайшему пути робот доезжает до цели без столкновений.
    SimConfig config;
    config.task.spawn_clearance = 0.18;
    config.task.max_steps = 2000;
    Simulator sim(config, test_maze());
    for (std::uint64_t seed = 1; seed <= 8; ++seed) {
        sim.reset(seed);
        const double optimal = sim.shortest_path_length();
        Vec2 previous = sim.robot_state().pose.position();
        double travelled = 0.0;
        StepResult r;
        while (!r.terminated && !r.truncated) {
            r = sim.step(pure_pursuit(sim));
            travelled += distance(previous, sim.robot_state().pose.position());
            previous = sim.robot_state().pose.position();
        }
        CHECK(r.success);
        CHECK(!r.collision);
        CHECK(travelled < 1.15 * optimal + 0.5);
    }
}

TEST(vec_simulator_is_independent_of_thread_count) {
    SimConfig config;
    config.lidar.num_rays = 90;
    config.task.max_steps = 60;  // короткие эпизоды: проверяем и автоперезапуск
    const std::vector<std::shared_ptr<const World>> worlds{test_maze(), std::make_shared<const World>(World::cluttered(7.0, 7.0, 0.05, 5, 4, 2))};
    const int envs = 11;
    VecSimulator one(config, worlds, envs, 1), many(config, worlds, envs, 4);
    one.reset(123);
    many.reset(123);
    CHECK(same_bytes(one.lidar(), many.lidar()));
    std::vector<double> actions(static_cast<std::size_t>(envs) * 2);
    int finished = 0;
    for (int i = 0; i < 400; ++i) {
        for (int e = 0; e < envs; ++e) {
            const WheelCommand c = wander(i, e);
            actions[2 * static_cast<std::size_t>(e)] = c.left;
            actions[2 * static_cast<std::size_t>(e) + 1] = c.right;
        }
        one.step(actions);
        many.step(actions);
        CHECK(same_bytes(one.lidar(), many.lidar()));
        CHECK(same_bytes(one.proprio(), many.proprio()));
        CHECK(same_bytes(one.state(), many.state()));
        CHECK(same_bytes(one.rewards(), many.rewards()));
        CHECK(same_bytes(one.terminated(), many.terminated()));
        CHECK(same_bytes(one.truncated(), many.truncated()));
        for (int e = 0; e < envs; ++e) finished += one.terminated()[static_cast<std::size_t>(e)] + one.truncated()[static_cast<std::size_t>(e)];
    }
    CHECK(finished > envs);  // эпизоды действительно завершались и перезапускались
    // После автоперезапуска среда находится на нулевом шаге нового эпизода.
    CHECK(one.env(0).step_count() < 60);
}

// ------------------------------------------------------------------------ IPC

TEST(shared_memory_matches_in_process_simulation) {
    const std::string name = "/roversim_test_" + std::to_string(::getpid());
    SimConfig config;
    config.lidar.num_rays = 180;
    const auto world = test_maze();

    std::atomic<bool> stop{false};
    ShmServer server(name, config, world);
    std::thread server_thread([&] { server.serve(stop); });
    {
        ShmClient client(name);
        ShmClient observer(name, 1.0, /*read_only=*/true);
        Simulator local(config, world);
        CHECK(client.header().num_rays == 180);
        CHECK_NEAR(client.header().control_dt, config.control_dt(), 0.0);

        std::uint64_t seed = 7;
        client.reset(seed);
        local.reset(seed);
        CHECK(same_bytes(client.lidar(), local.lidar()));
        for (int i = 0; i < 600; ++i) {
            const WheelCommand c = wander(i);
            const StepResult remote = client.step(c.left, c.right);
            const StepResult expected = local.step(c);
            CHECK(remote.reward == expected.reward);
            CHECK(remote.terminated == expected.terminated && remote.truncated == expected.truncated);
            CHECK(remote.collision == expected.collision && remote.success == expected.success);
            CHECK(same_bytes(client.lidar(), local.lidar()));
            CHECK(same_bytes(client.proprio(), local.proprio()));
            CHECK(same_bytes(client.state(), local.state()));
            CHECK(client.step_count() == local.step_count());
            if (remote.terminated || remote.truncated) {
                client.reset(++seed);
                local.reset(seed);
            }
        }

        // Время выполнения команды сервером вложено во время запроса у клиента.
        const auto request_start = std::chrono::steady_clock::now();
        client.step(1.0, 1.0);
        const double round_trip = std::chrono::duration<double>(std::chrono::steady_clock::now() - request_start).count();
        local.step({1.0, 1.0});
        CHECK(server.last_command_seconds() > 0.0);
        CHECK(server.last_command_seconds() <= round_trip);

        // Время активного ожидания меняется на ходу и на результат не влияет:
        // без спина стороны будят друг друга через futex.
        CHECK_NEAR(server.spin_time(), kShmDefaultSpinTime, 1e-12);
        server.set_spin_time(0.0);
        client.set_spin_time(0.0);
        for (int i = 0; i < 50; ++i) {
            const WheelCommand c = wander(i);
            const StepResult remote = client.step(c.left, c.right);
            const StepResult expected = local.step(c);
            CHECK(remote.reward == expected.reward);
            CHECK(same_bytes(client.lidar(), local.lidar()));
            if (remote.terminated || remote.truncated) {
                client.reset(++seed);
                local.reset(seed);
            }
        }
        server.set_spin_time(kShmDefaultSpinTime);
        client.set_spin_time(kShmDefaultSpinTime);
        bool bad_spin = false;
        try {
            server.set_spin_time(-1.0);
        } catch (const std::invalid_argument&) {
            bad_spin = true;
        }
        CHECK(bad_spin);

        // Наблюдатель «только чтение» получает согласованный снимок через seqlock.
        std::vector<float> lidar(180), proprio(obs::Dim);
        std::vector<double> state(gt::Dim);
        CHECK(observer.snapshot(lidar, proprio, state));
        CHECK(same_bytes(std::span<const float>(lidar), local.lidar()));
        CHECK(same_bytes(std::span<const double>(state), local.state()));
        bool refused = false;
        try {
            observer.step(1.0, 1.0);
        } catch (const std::logic_error&) {
            refused = true;
        }
        CHECK(refused);

        // Некорректная команда не роняет сервер: клиент получает исключение и работает дальше.
        bool rejected = false;
        try {
            client.reset(1, Pose2{0.01, 0.01, 0.0}, Vec2{3.0, 3.0});
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        CHECK(rejected);
        client.reset(11);
        local.reset(11);
        CHECK(same_bytes(client.lidar(), local.lidar()));
        client.close_server();
    }
    server_thread.join();

    // Сегмент занят живым сервером — второй с тем же именем не создаётся.
    bool busy = false;
    try {
        ShmServer duplicate(name, config, world);
    } catch (const std::runtime_error&) {
        busy = true;
    }
    CHECK(busy);
}

// --------------------------------------------------------------- визуализация

TEST(renderer_draws_scene) {
    Simulator sim(SimConfig{}, test_maze());
    sim.reset(4);
    viz::Renderer renderer(200, 160);
    const viz::Image& image = renderer.render(sim);
    CHECK(image.width == 200 && image.height == 160);
    CHECK(image.rgb.size() == 200U * 160U * 3U);
    // В кадре есть и стены, и пол, и робот: много разных цветов.
    std::vector<std::uint32_t> colors;
    for (std::size_t i = 0; i < image.rgb.size(); i += 3) {
        colors.push_back((std::uint32_t{image.rgb[i]} << 16) | (std::uint32_t{image.rgb[i + 1]} << 8) | image.rgb[i + 2]);
    }
    std::sort(colors.begin(), colors.end());
    colors.erase(std::unique(colors.begin(), colors.end()), colors.end());
    CHECK(colors.size() > 8);
    CHECK(std::binary_search(colors.begin(), colors.end(), std::uint32_t{0xC4CAD4}));  // цвет стен

    const std::string frame = viz::to_ansi(image);
    CHECK(std::count(frame.begin(), frame.end(), '\n') == 80);  // две строки пикселей на строку текста
    CHECK(frame.find("\xE2\x96\x80") != std::string::npos);
}

int connect_local(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const timeval timeout{2, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

std::string read_until(int fd, std::string_view marker, std::size_t at_least = 0) {
    std::string data;
    char buffer[4096];
    while (data.find(marker) == std::string::npos || data.size() < at_least) {
        const ssize_t got = ::recv(fd, buffer, sizeof(buffer), 0);
        if (got <= 0) break;
        data.append(buffer, static_cast<std::size_t>(got));
    }
    return data;
}

TEST(web_viewer_serves_page_and_websocket) {
    Simulator sim(SimConfig{}, test_maze());
    sim.reset(4);
    viz::WebViewer viewer(0);  // любой свободный порт
    CHECK(viewer.port() != 0);

    // Обычный HTTP-запрос: страница просмотрщика.
    int fd = connect_local(viewer.port());
    CHECK(fd >= 0);
    const std::string get = "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";
    CHECK(::send(fd, get.data(), get.size(), 0) == static_cast<ssize_t>(get.size()));
    const std::string page = read_until(fd, "</html>");
    CHECK(page.starts_with("HTTP/1.1 200 OK"));
    CHECK(page.find("<canvas") != std::string::npos);
    ::close(fd);

    // Рукопожатие WebSocket; ключ и ответ — пример из RFC 6455.
    fd = connect_local(viewer.port());
    CHECK(fd >= 0);
    const std::string upgrade = "GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                                "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
    CHECK(::send(fd, upgrade.data(), upgrade.size(), 0) == static_cast<ssize_t>(upgrade.size()));
    const std::string handshake = read_until(fd, "\r\n\r\n");
    CHECK(handshake.starts_with("HTTP/1.1 101"));
    CHECK(handshake.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") != std::string::npos);

    // Публикуем кадры, пока сервер не увидит зрителя, затем читаем сообщения.
    for (int i = 0; i < 400 && viewer.num_clients() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(viewer.num_clients() == 1);
    viewer.publish(sim, true);
    const std::size_t world_bytes = 32 + sim.world().cells().size();
    const std::string stream = read_until(fd, "", 4 + world_bytes + 4 + 96 + sim.lidar().size_bytes());
    // Первое сообщение — карта: бинарный кадр (0x82) с 16-битной длиной и типом 1.
    CHECK(stream.size() >= 4 + world_bytes + 4 + 96);
    CHECK(static_cast<unsigned char>(stream[0]) == 0x82 && static_cast<unsigned char>(stream[1]) == 126);
    const std::size_t length = (static_cast<unsigned char>(stream[2]) << 8) | static_cast<unsigned char>(stream[3]);
    CHECK(length == world_bytes);
    CHECK(stream[4] == 1);
    // Второе — кадр состояния (тип 2) с числом лучей в заголовке.
    const std::size_t second = 4 + world_bytes;
    CHECK(static_cast<unsigned char>(stream[second]) == 0x82);
    CHECK(stream[second + 4] == 2);
    std::uint32_t rays = 0;
    std::memcpy(&rays, stream.data() + second + 8, sizeof(rays));
    CHECK(rays == sim.lidar().size());

    // Ручное управление: браузер шлёт маскированный кадр с двумя float (v, w).
    double v = 0.0, w = 0.0;
    CHECK(!viewer.teleop(v, w));
    const float command[2] = {1.0F, -0.5F};
    const unsigned char mask[4] = {0x12, 0x34, 0x56, 0x78};
    std::string frame{static_cast<char>(0x82), static_cast<char>(0x80 | 8)};
    frame.append(reinterpret_cast<const char*>(mask), 4);
    for (std::size_t i = 0; i < 8; ++i) frame.push_back(static_cast<char>(reinterpret_cast<const unsigned char*>(command)[i] ^ mask[i & 3]));
    CHECK(::send(fd, frame.data(), frame.size(), 0) == static_cast<ssize_t>(frame.size()));
    bool received = false;
    for (int i = 0; i < 200 && !received; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        received = viewer.teleop(v, w);
    }
    CHECK(received);
    CHECK_NEAR(v, 1.0, 0.0);
    CHECK_NEAR(w, -0.5, 0.0);
    ::close(fd);
}


// ------------------------------------------------- подвижные препятствия и маршрут

TEST(moving_obstacles_bounce_and_are_seen_by_lidar) {
    SimConfig config = quiet_config();
    config.obstacles.count = 4;
    const auto arena = std::make_shared<const World>(World::arena(8.0, 8.0, 0.05));
    Simulator sim(config, arena);
    sim.reset(3, Pose2{4.0, 4.0, 0.0}, Vec2{1.0, 1.0});
    CHECK(sim.obstacles().size() == 4);
    std::vector<Vec2> start;
    for (const DynamicObstacle& o : sim.obstacles()) start.push_back(o.position);

    const Lidar2D& lidar = sim.lidar_model();
    double worst_lidar = 0.0;
    for (int i = 0; i < 400; ++i) {
        sim.step({0.0, 0.0});
        const Vec2 robot = sim.robot_state().pose.position();
        for (const DynamicObstacle& o : sim.obstacles()) {
            // Препятствие не выходит за стены, движется с заданной скоростью...
            CHECK(sim.world().is_free(o.position, o.radius - 1e-6));
            const double speed = norm(o.velocity);
            CHECK(speed >= config.obstacles.speed_min - 1e-9 && speed <= config.obstacles.speed_max + 1e-9);
            // ...и лидар видит его: луч, направленный в центр, упирается в ближнюю точку круга.
            const Vec2 rel = o.position - robot;
            const double dist = norm(rel), bearing = std::atan2(rel.y, rel.x) - sim.robot_state().pose.theta;
            const int ray = static_cast<int>(std::lround((wrap_angle(bearing) - lidar.angle_min()) / lidar.angle_increment())) % lidar.num_rays();
            bool occluded = false;  // другое препятствие ближе на том же луче
            for (const DynamicObstacle& other : sim.obstacles()) occluded = occluded || (&other != &o && norm(other.position - robot) < dist);
            if (!occluded) worst_lidar = std::max(worst_lidar, std::abs(sim.lidar()[static_cast<std::size_t>(ray)] - (dist - o.radius)));
        }
    }
    CHECK_NEAR(worst_lidar, 0.0, 0.01);  // погрешность — только дискретность луча по углу
    for (std::size_t i = 0; i < start.size(); ++i) CHECK(distance(start[i], sim.obstacles()[i].position) > 0.5);

    // Стоящего робота препятствия не толкают: они отскакивают при касании.
    CHECK_NEAR(distance(sim.robot_state().pose.position(), Vec2{4.0, 4.0}), 0.0, 1e-9);

    // С препятствиями симуляция остаётся детерминированной.
    const auto run = [&](std::uint64_t seed) {
        SimConfig noisy;
        noisy.obstacles.count = 4;
        noisy.task.terminate_on_collision = false;
        Simulator again(noisy, arena);
        again.reset(seed);
        std::uint64_t hash = 0xCBF29CE484222325ULL;
        for (int i = 0; i < 400; ++i) {
            again.step(wander(i));
            hash = hash_observation(hash, again);
        }
        return hash;
    };
    CHECK(run(1) == run(1));
    CHECK(run(1) != run(2));
}

TEST(robot_stops_at_moving_obstacle_without_overlap) {
    // Робот едет прямо на препятствия. Контакт фиксируется как столкновение, а
    // тела не проникают друг в друга.
    SimConfig config = quiet_config();
    config.obstacles.count = 3;
    const auto arena = std::make_shared<const World>(World::arena(6.0, 6.0, 0.05));
    Simulator sim(config, arena);
    int collisions = 0;
    double deepest = 0.0;
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        sim.reset(seed, Pose2{3.0, 3.0, 0.0}, Vec2{1.0, 1.0});
        for (int i = 0; i < 600; ++i) {
            // Преследование ближайшего препятствия.
            const Pose2 pose = sim.robot_state().pose;
            Vec2 target = sim.obstacles()[0].position;
            for (const DynamicObstacle& o : sim.obstacles()) {
                if (distance(o.position, pose.position()) < distance(target, pose.position())) target = o.position;
            }
            const Vec2 local = rotate_inv(target - pose.position(), std::cos(pose.theta), std::sin(pose.theta));
            const double bearing = std::atan2(local.y, local.x);
            const StepResult r = sim.step(sim.drive().twist_to_wheels(std::abs(bearing) < 0.5 ? 0.8 : 0.0, 3.0 * bearing));
            collisions += r.collision ? 1 : 0;
            for (const DynamicObstacle& o : sim.obstacles()) {
                const double gap = distance(o.position, sim.robot_state().pose.position()) - o.radius - config.robot.radius;
                deepest = std::min(deepest, gap);
            }
        }
    }
    CHECK(collisions > 0);
    CHECK(deepest > -1e-3);
}

TEST(path_hint_follows_shortest_path) {
    // На пустой арене кратчайший путь — прямая: подсказка совпадает с направлением на цель.
    SimConfig config = quiet_config();
    Simulator open(config, open_arena());
    open.reset(1, Pose2{5.0, 5.0, 0.7}, Vec2{12.0, 9.0});
    CHECK_NEAR(open.proprio()[obs::PathCos], open.proprio()[obs::GoalCos], 0.05);
    CHECK_NEAR(open.proprio()[obs::PathSin], open.proprio()[obs::GoalSin], 0.05);

    // В лабиринте простейший регулятор «рули по подсказке» доезжает до цели без
    // столкновений — подсказка действительно ведёт в обход стен.
    config.task.max_steps = 3000;
    config.task.spawn_clearance = 0.18;
    config.task.terminate_on_collision = true;
    config.task.lookahead = 0.5;
    Simulator sim(config, test_maze());
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        sim.reset(seed);
        StepResult r;
        while (!r.terminated && !r.truncated) {
            const double bearing = std::atan2(sim.proprio()[obs::PathSin], sim.proprio()[obs::PathCos]);
            const double v = std::abs(bearing) < 0.6 ? 0.5 * std::cos(bearing) : 0.0;
            r = sim.step(sim.drive().twist_to_wheels(v, 2.5 * bearing));
        }
        CHECK(r.success);
        CHECK(!r.collision);
    }
}

// ------------------------------------------------------------------ навигация

TEST(occupancy_grid_traces_rays_with_bresenham) {
    nav::MapConfig config;
    config.size_x = config.size_y = 4.0;
    config.surface_clearance = 0.0;  // здесь проверяется сама трассировка, без «хвоста» у поверхности
    nav::OccupancyGrid map(config, {2.0, 2.0});
    CHECK(map.width() == 80 && map.height() == 80);
    const Vec2 dirs[2] = {{1.0, 0.0}, {std::sqrt(0.5), std::sqrt(0.5)}};
    const float ranges[2] = {1.0F, 10.0F};  // первый луч упирается в стену, второй уходит в пустоту
    const Pose2 sensor{1.025, 1.025, 0.0};  // центр клетки (20, 20)
    map.integrate_scan(sensor, dirs, ranges, 10.0);

    // Горизонтальный луч: клетки 20..39 свободны, клетка 40 занята, дальше неизвестно.
    for (int ix = 20; ix < 40; ++ix) CHECK_NEAR(map.log_odds(ix, 20), -config.log_odds_miss, 1e-6);
    CHECK_NEAR(map.log_odds(40, 20), config.log_odds_hit, 1e-6);
    CHECK(map.log_odds(41, 20) == 0.0F);
    CHECK(map.state(40, 20) == nav::CellState::Occupied);
    CHECK(map.state(41, 20) == nav::CellState::Unknown);
    // Диагональный луч без отражения: клетки на диагонали свободны, занятых нет.
    for (int k = 1; k < 55; ++k) CHECK_NEAR(map.log_odds(20 + k, 20 + k), -config.log_odds_miss, 1e-6);
    CHECK(map.log_odds(30, 31) == 0.0F);  // соседи диагонали не затронуты
    // Каждая клетка обновляется раз за скан: общая стартовая клетка получила одно «свободно».
    CHECK_NEAR(map.log_odds(20, 20), -config.log_odds_miss, 1e-6);

    // Многократные наблюдения: значения насыщаются на границах.
    for (int i = 0; i < 20; ++i) map.integrate_scan(sensor, dirs, ranges, 10.0);
    CHECK_NEAR(map.log_odds(40, 20), config.log_odds_max, 1e-6);
    CHECK_NEAR(map.log_odds(30, 20), config.log_odds_min, 1e-6);
    CHECK_NEAR(map.probability(40, 20), 1.0 / (1.0 + std::exp(-config.log_odds_max)), 1e-9);
    // Интерполированное значение: +1 в центре занятой клетки, -1 в центре свободной.
    CHECK_NEAR(map.occupancy_value(map.cell_center(40, 20)), 1.0, 1e-6);
    CHECK_NEAR(map.occupancy_value(map.cell_center(30, 20)), -1.0, 1e-6);

    // Стена «забывается», но не сразу. Клетка помнит свои попадания (их было 21), и
    // первые наблюдения «свободно» лишь гасят их; когда память исчерпана, оценка
    // занятости начинает снижаться, и клетка освобождается.
    const float through[2] = {3.0F, 10.0F};
    for (int i = 0; i < 20; ++i) map.integrate_scan(sensor, dirs, through, 10.0);
    CHECK(map.state(40, 20) == nav::CellState::Occupied);
    for (int i = 0; i < 20; ++i) map.integrate_scan(sensor, dirs, through, 10.0);
    CHECK(map.state(40, 20) == nav::CellState::Free);
}

TEST(mapping_with_known_poses_reproduces_world) {
    // Робот едет по лабиринту; карта строится по истинным позам и скану без шума.
    // Проверяется согласие карты с истинным миром.
    SimConfig config = quiet_config();
    config.task.spawn_clearance = 0.18;
    const auto world = test_maze();
    Simulator sim(config, world);
    sim.reset(2);
    const nav::ScanGeometry scan = nav::scan_geometry(sim.lidar_model());
    nav::SlamConfig slam_config;
    slam_config.scan_matching = false;
    nav::GridSlam slam(slam_config, scan, sim.robot_state().pose);
    for (int i = 0; i < 1500; ++i) {
        slam.update(sim.robot_state().pose, sim.lidar());  // истинная поза вместо одометрии
        const StepResult r = sim.step(pure_pursuit(sim));
        if (r.terminated) sim.reset(static_cast<std::uint64_t>(10 + i));
    }
    const nav::OccupancyGrid& map = slam.map();
    int occupied = 0, occupied_ok = 0, free_cells = 0, free_ok = 0;
    for (int iy = 0; iy < map.height(); ++iy) {
        for (int ix = 0; ix < map.width(); ++ix) {
            const nav::CellState state = map.state(ix, iy);
            if (state == nav::CellState::Unknown) continue;
            const Vec2 p = map.cell_center(ix, iy);
            const int wx = static_cast<int>(std::floor(p.x / world->cell_size())), wy = static_cast<int>(std::floor(p.y / world->cell_size()));
            if (state == nav::CellState::Free) {
                ++free_cells;
                free_ok += world->occupied(wx, wy) ? 0 : 1;
            } else {
                ++occupied;
                // Занятая клетка карты — стена или клетка вплотную к ней (отражение на границе клеток).
                bool near_wall = false;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) near_wall = near_wall || world->occupied(wx + dx, wy + dy);
                }
                occupied_ok += near_wall ? 1 : 0;
            }
        }
    }
    CHECK(free_cells > 5000 && occupied > 500);
    CHECK(free_ok >= 0.995 * free_cells);
    CHECK(occupied_ok == occupied);
}

TEST(distance_transform_is_exact) {
    Rng rng(5);
    for (int trial = 0; trial < 5; ++trial) {
        const int w = 20 + static_cast<int>(rng.below(30)), h = 20 + static_cast<int>(rng.below(30));
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(w) * h, 0);
        for (auto& cell : mask) cell = rng.uniform() < 0.03 ? 1 : 0;
        mask[rng.below(mask.size())] = 1;  // хотя бы одна занятая клетка
        std::vector<float> distance_field;
        nav::distance_transform(mask, w, h, distance_field);
        double worst = 0.0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double best = 1e18;
                for (int oy = 0; oy < h; ++oy) {
                    for (int ox = 0; ox < w; ++ox) {
                        if (mask[static_cast<std::size_t>(oy) * w + ox] != 0) best = std::min(best, std::hypot(x - ox, y - oy));
                    }
                }
                worst = std::max(worst, std::abs(best - distance_field[static_cast<std::size_t>(y) * w + x]));
            }
        }
        CHECK_NEAR(worst, 0.0, 1e-4);
    }
}

double path_length(const std::vector<Vec2>& path) {
    double length = 0.0;
    for (std::size_t i = 1; i < path.size(); ++i) length += distance(path[i - 1], path[i]);
    return length;
}

TEST(planners_find_short_safe_paths) {
    SimConfig config = quiet_config();
    config.lidar.num_rays = 8;
    const auto world = test_maze();
    Simulator sim(config, world);

    nav::PlannerConfig theta_config, astar_config, raw_config;
    astar_config.algorithm = raw_config.algorithm = nav::PlannerConfig::Algorithm::AStar;
    raw_config.smooth = false;
    nav::GridPlanner theta(theta_config), astar(astar_config), raw(raw_config);
    theta.set_map(*world);
    astar.set_map(*world);
    raw.set_map(*world);

    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        sim.reset(seed);
        const Vec2 start = sim.robot_state().pose.position(), goal = sim.goal();
        std::vector<Vec2> theta_path, astar_path, raw_path;
        CHECK(theta.plan(start, goal, theta_path));
        CHECK(astar.plan(start, goal, astar_path));
        CHECK(raw.plan(start, goal, raw_path));

        // Пути начинаются и кончаются где нужно и нигде не подходят к стенам ближе радиуса робота.
        for (const std::vector<Vec2>* path : {&theta_path, &astar_path, &raw_path}) {
            CHECK(distance(path->front(), start) == 0.0 && distance(path->back(), goal) == 0.0);
            for (std::size_t i = 1; i < path->size(); ++i) {
                const double length = distance((*path)[i - 1], (*path)[i]);
                const int samples = std::max(1, static_cast<int>(length / 0.02));
                for (int k = 0; k <= samples; ++k) {
                    const Vec2 p = (*path)[i - 1] + ((*path)[i] - (*path)[i - 1]) * (static_cast<double>(k) / samples);
                    CHECK(world->is_free(p, config.robot.radius));
                }
            }
        }
        // Путь «под любым углом» не длиннее сеточного и состоит из меньшего числа отрезков.
        const double shortest = sim.shortest_path_length();
        CHECK(path_length(theta_path) <= path_length(raw_path) + 1e-6);
        CHECK(theta_path.size() <= raw_path.size());
        // Оба близки к кратчайшему (он проложен вплотную к стенам, эти — с запасом).
        CHECK(path_length(theta_path) < 1.25 * shortest + 0.5);
        CHECK(path_length(raw_path) < 1.25 * shortest + 0.5);
        CHECK(path_length(theta_path) > 0.85 * shortest - 0.5);
    }

    // Недостижимая цель: комната, полностью отделённая стеной.
    World split(80, 40, 0.1);
    for (int iy = 0; iy < 40; ++iy) split.set_cell(40, iy, true);
    split.finalize();
    nav::GridPlanner blocked(theta_config);
    blocked.set_map(split);
    std::vector<Vec2> none;
    CHECK(!blocked.plan({1.0, 2.0}, {7.0, 2.0}, none));
    CHECK(none.empty());
    CHECK(blocked.plan({1.0, 2.0}, {3.0, 3.0}, none));
    CHECK_NEAR(blocked.clearance({2.0, 2.0}), 2.0 - 0.025, 0.051);  // расстояние до стены x = 4.0
    CHECK(blocked.line_free({1.0, 1.0}, {3.0, 3.0}));
    CHECK(!blocked.line_free({1.0, 1.0}, {7.0, 1.0}));
}

TEST(slam_corrects_odometry_drift) {
    // Длинный проезд по лабиринту с шумной одометрией. Оценка SLAM должна
    // оставаться вблизи истины, а «сырая» одометрия — заметно уплыть.
    SimConfig config;
    config.task.max_steps = 0;
    config.task.terminate_on_collision = false;
    config.task.spawn_clearance = 0.18;
    const auto world = test_maze();
    Simulator sim(config, world);
    sim.reset(4);
    const nav::ScanGeometry scan = nav::scan_geometry(sim.lidar_model());
    nav::GridSlam slam(nav::SlamConfig{}, scan, sim.odom_pose());

    // Цели меняются сами, а эпизод один: одометрия копит ошибку непрерывно.
    const Vec2 waypoints[] = {{0.8, 0.8}, {6.4, 6.4}, {0.8, 6.4}, {6.4, 0.8}, {3.6, 3.6}, {0.8, 0.8}};
    nav::PlannerConfig planner_config;
    nav::GridPlanner planner(planner_config);
    planner.set_map(*world);
    double worst_slam = 0.0;
    std::size_t target = 0;
    std::vector<Vec2> path;
    for (int i = 0; i < 4000 && target < std::size(waypoints); ++i) {
        const Pose2 truth = sim.robot_state().pose;
        slam.update(sim.odom_pose(), sim.lidar());
        worst_slam = std::max(worst_slam, distance(slam.pose().position(), truth.position()));
        if (distance(truth.position(), waypoints[target]) < 0.35) ++target;
        if (target == std::size(waypoints)) break;
        // Водитель с истинной позой: путь по истинной карте и погоня за точкой впереди.
        planner.plan(truth.position(), waypoints[target], path);
        Vec2 carrot = waypoints[target];
        double walked = 0.0;
        for (std::size_t k = 1; k < path.size(); ++k) {
            const double segment = distance(path[k - 1], path[k]);
            if (walked + segment >= 0.4) {
                carrot = path[k - 1] + (path[k] - path[k - 1]) * ((0.4 - walked) / segment);
                break;
            }
            walked += segment;
        }
        const Vec2 local = rotate_inv(carrot - truth.position(), std::cos(truth.theta), std::sin(truth.theta));
        const double bearing = std::atan2(local.y, local.x);
        sim.step(sim.drive().twist_to_wheels(std::abs(bearing) < 0.7 ? 0.6 * std::cos(bearing) : 0.0, 2.5 * bearing));
    }
    CHECK(target == std::size(waypoints));  // маршрут пройден целиком
    const Pose2 truth = sim.robot_state().pose;
    const double odom_error = distance(sim.odom_pose().position(), truth.position());
    const double slam_error = distance(slam.pose().position(), truth.position());
    CHECK(odom_error > 0.10);          // одометрия действительно уплыла
    CHECK(slam_error < 0.06);          // а SLAM — нет
    CHECK(worst_slam < 0.10);          // и не уплывал по дороге
    CHECK(slam_error < 0.4 * odom_error);
    CHECK(slam.match_score() > 0.5);   // скан хорошо ложится на карту
}

TEST(dwa_respects_dynamics_and_obstacles) {
    nav::DwaConfig config;
    nav::DwaPlanner dwa(config);
    const std::vector<Vec2> empty;

    // Свободное пространство, цель впереди: разгон в пределах динамического окна, почти без поворота.
    nav::DwaResult r = dwa.compute(empty, {}, {2.0, 0.0}, 0.0, 0.0);
    CHECK(r.feasible && r.admissible == config.speed_samples * config.yaw_samples);
    CHECK(r.v > 0.0 && r.v <= config.max_accel * config.window_time + 1e-9);
    CHECK_NEAR(r.w, 0.0, 0.3);
    // Цель слева — поворот влево, справа — вправо.
    CHECK(dwa.compute(empty, {}, {1.0, 1.5}, 0.3, 0.0).w > 0.2);
    CHECK(dwa.compute(empty, {}, {1.0, -1.5}, 0.3, 0.0).w < -0.2);
    // Скорость не превышает предельную и не уходит за окно.
    r = dwa.compute(empty, {}, {5.0, 0.0}, config.max_speed, 0.0);
    CHECK(r.v <= config.max_speed + 1e-9 && r.v >= config.max_speed - config.max_accel * config.window_time - 1e-9);

    // Стена поперёк пути в 0.45 м перед роботом (край робота — в 0.25 м от неё).
    std::vector<Vec2> wall;
    for (double y = -1.5; y <= 1.5; y += 0.03) wall.push_back({0.45, y});
    r = dwa.compute(wall, {}, {2.0, 0.0}, 0.5, 0.0);
    CHECK(r.feasible);
    CHECK(r.v < 0.5);  // торможение
    // Выбранная траектория не подходит к стене ближе запаса.
    for (const Vec2& p : dwa.trajectory()) CHECK(0.45 - p.x - config.robot_radius > config.safety_margin - 0.04);
    // Из положения у самой стены приближаться нельзя вовсе.
    std::vector<Vec2> touching;
    for (double y = -1.5; y <= 1.5; y += 0.03) touching.push_back({0.22, y});
    r = dwa.compute(touching, {}, {2.0, 0.0}, 0.0, 0.0);
    CHECK(!r.feasible || r.v <= 1e-9);

    // Препятствие едет навстречу: с разрешённым задним ходом робот отступает,
    // без него — по крайней мере не едет вперёд.
    const nav::MovingObstacle incoming[] = {{{1.0, 0.0}, {-0.5, 0.0}, 0.2}};
    r = dwa.compute(empty, incoming, {2.0, 0.0}, 0.0, 0.0, /*allow_reverse=*/true);
    CHECK(r.feasible && r.v < 0.0);
    // Без заднего хода остаётся объезд: робот сворачивает, и на выбранной
    // траектории нет встречи с препятствием (с учётом его движения).
    r = dwa.compute(empty, incoming, {2.0, 0.0}, 0.0, 0.0, /*allow_reverse=*/false);
    CHECK(r.feasible && r.v >= 0.0 && std::abs(r.w) > 0.3);
    for (std::size_t k = 0; k < dwa.trajectory().size(); ++k) {
        const double time = (static_cast<double>(k) + 1.0) * config.time_step;
        const Vec2 obstacle = incoming[0].position + incoming[0].velocity * time;
        CHECK(distance(dwa.trajectory()[k], obstacle) > incoming[0].radius + config.robot_radius);
    }
    // Препятствие едет поперёк и освободит дорогу: неподвижным оно перекрывало бы путь.
    const nav::MovingObstacle crossing[] = {{{0.9, 0.0}, {0.0, 0.5}, 0.2}};
    const nav::MovingObstacle parked[] = {{{0.9, 0.0}, {0.0, 0.0}, 0.2}};
    CHECK(dwa.compute(empty, crossing, {2.5, 0.0}, 0.4, 0.0).score > dwa.compute(empty, parked, {2.5, 0.0}, 0.4, 0.0).score);
}

TEST(dwa_keeps_wall_margin_when_touched_by_moving_obstacle) {
    // Робот зажат: сзади вплотную подвижное препятствие (оно уже отъезжает),
    // впереди, в 8 см от корпуса, стена, за стеной — цель. Близость к препятствию
    // не должна отменять запас до стены: с общим порогом допустимости на оба вида
    // препятствий робот, отъезжая от одного, упирался в другое.
    nav::DwaConfig config;
    nav::DwaPlanner dwa(config);
    const double wall_x = config.robot_radius + 0.08;
    std::vector<Vec2> wall;
    for (double y = -1.5; y <= 1.5; y += 0.03) wall.push_back({wall_x, y});
    // Лидар видит и само препятствие: точки его ближней поверхности тоже в скане.
    const nav::MovingObstacle behind[] = {{{-0.40, 0.0}, {-0.3, 0.0}, 0.2}};
    std::vector<Vec2> scan = wall;
    for (double a = -0.6; a <= 0.6; a += 0.1) {
        scan.push_back(behind[0].position + Vec2{std::cos(a), std::sin(a)} * behind[0].radius);
    }
    // Сетка локального поля расстояний — 5 см, отсюда допуск в проверках зазора.
    const auto keeps_margin = [&] {
        for (const Vec2& p : dwa.trajectory()) {
            if (wall_x - p.x - config.robot_radius < config.safety_margin - 0.04) return false;
        }
        return true;
    };
    for (const bool reverse : {false, true}) {
        const nav::DwaResult r = dwa.compute(scan, behind, {2.0, 0.0}, 0.0, 0.0, reverse);
        CHECK(r.feasible);
        CHECK(keeps_margin());
        CHECK(r.v < 0.15);  // к стене робот не разгоняется
    }
    // Та же стена без препятствия за спиной: ограничение то же самое.
    const nav::DwaResult alone = dwa.compute(wall, {}, {2.0, 0.0}, 0.0, 0.0);
    CHECK(alone.feasible);
    CHECK(keeps_margin());
    CHECK(alone.v < 0.15);
    // Сблизиться с самим препятствием тоже нельзя: если оно стоит, задний ход закрыт.
    const nav::MovingObstacle standing[] = {{{-0.40, 0.0}, {0.0, 0.0}, 0.2}};
    const nav::DwaResult r = dwa.compute(scan, standing, {-2.0, 0.0}, 0.0, 0.0, /*allow_reverse=*/true);
    CHECK(!r.feasible || r.v >= -1e-9);
}

TEST(tracker_estimates_obstacle_velocity) {
    // Робот стоит; мимо ездят препятствия. Навигатор должен выделить их из скана
    // (сравнением с картой) и оценить скорости.
    SimConfig config;
    config.obstacles.count = 2;
    config.task.max_steps = 0;
    config.task.terminate_on_collision = false;
    const auto arena = std::make_shared<const World>(World::arena(7.0, 7.0, 0.05));
    Simulator sim(config, arena);
    sim.reset(5, Pose2{3.5, 3.5, 0.0}, Vec2{1.0, 1.0});
    nav::Navigator navigator(nav::navigator_config(config.robot, config.control_dt()), nav::scan_geometry(sim.lidar_model()), sim.odom_pose());

    int matched = 0, checks = 0;
    double velocity_error = 0.0, position_error = 0.0;
    for (int i = 0; i < 400; ++i) {
        navigator.update(sim.lidar(), sim.odom_pose(), 0.0, 0.0);  // цели нет: только восприятие
        sim.step({0.0, 0.0});
        if (i < 60) continue;  // время на разведку пола и подтверждение треков
        for (const DynamicObstacle& o : sim.obstacles()) {
            ++checks;
            for (const nav::Track& track : navigator.tracks()) {
                if (!track.moving || distance(track.position, o.position) > 0.3) continue;
                ++matched;
                velocity_error += distance(track.velocity, o.velocity);
                position_error += distance(track.position, o.position);
                break;
            }
        }
    }
    CHECK(matched > 0.7 * checks);  // препятствия сопровождаются почти всё время
    CHECK(velocity_error / matched < 0.10);
    CHECK(position_error / matched < 0.12);
    // Подвижные объекты в статическую карту не попадают: в центре арены пусто.
    int stray = 0;
    const nav::OccupancyGrid& map = navigator.map();
    for (int iy = 0; iy < map.height(); ++iy) {
        for (int ix = 0; ix < map.width(); ++ix) {
            const Vec2 p = map.cell_center(ix, iy);
            const bool interior = p.x > 0.4 && p.x < 6.6 && p.y > 0.4 && p.y < 6.6;
            stray += interior && map.state(ix, iy) == nav::CellState::Occupied ? 1 : 0;
        }
    }
    CHECK(stray < 12);
}

TEST(tracker_tells_moving_objects_from_new_static_ones) {
    // Синтетический скан: вокруг стены на 3 м, а в секторе из пяти лучей — объект
    // ближе. Отметки «отражение пришло с пола» задаются вручную.
    nav::ScanGeometry scan;
    scan.num_rays = 360;
    scan.angle_min = -kPi;
    scan.angle_increment = kTwoPi / 360;
    scan.range_max = 10.0;
    const nav::TrackerConfig config;
    const double dt = 0.05;
    const auto make_scan = [](double range, std::vector<float>& ranges, std::vector<std::uint8_t>& transient) {
        ranges.assign(360, 3.0F);
        transient.assign(360, 0);
        for (std::size_t i = 178; i <= 182; ++i) {  // луч 180 смотрит вдоль оси x
            ranges[i] = static_cast<float>(range);
            transient[i] = 1;
        }
    };
    std::vector<float> ranges;
    std::vector<std::uint8_t> transient, flags;

    // 1. Неподвижный предмет. По умолчанию он числится подвижным с момента
    //    подтверждения и до «парковки»; с правилом смещения (confirm_travel > 0) —
    //    не числится вовсе. В обоих случаях спустя park_time его лучи отдаются
    //    карте с отметкой Settled.
    for (const double confirm_travel : {0.0, 0.15}) {
        nav::TrackerConfig variant = config;
        variant.confirm_travel = confirm_travel;
        nav::ObstacleTracker tracker(variant);
        bool ever_moving = false;
        int settled_at = -1;
        for (int k = 0; k < 80; ++k) {
            make_scan(1.52, ranges, transient);
            tracker.update(Pose2{}, scan, ranges, transient, dt, flags);
            for (const nav::Track& track : tracker.tracks()) ever_moving = ever_moving || track.moving;
            if (settled_at < 0 && flags[180] == nav::ray_flag::Settled) settled_at = k;
            if (settled_at < 0) CHECK(flags[180] == nav::ray_flag::Dynamic);  // до тех пор в карту не идёт
            CHECK(flags[90] == nav::ray_flag::Static);                        // лучи стен трекер не трогает
        }
        CHECK(tracker.tracks().size() == 1);
        CHECK(ever_moving == (confirm_travel <= 0.0));
        CHECK(!tracker.tracks()[0].moving);  // после «парковки» подвижным не числится
        CHECK(settled_at > 0);
        CHECK_NEAR(settled_at * dt, config.park_time, 0.2);
    }

    // 2. Объект приближается со скоростью 0.4 м/с. По умолчанию он подвижен со
    //    второго наблюдения; с правилом смещения — когда сместился на confirm_travel.
    //    Оценка скорости сходится к истинной.
    for (const double confirm_travel : {0.0, 0.15}) {
        nav::TrackerConfig variant = config;
        variant.confirm_travel = confirm_travel;
        nav::ObstacleTracker tracker(variant);
        int moving_at = -1;
        for (int k = 0; k < 60; ++k) {
            make_scan(2.8 - 0.4 * dt * k, ranges, transient);
            tracker.update(Pose2{}, scan, ranges, transient, dt, flags);
            CHECK(flags[180] == nav::ray_flag::Dynamic);
            if (moving_at < 0 && !tracker.tracks().empty() && tracker.tracks()[0].moving) moving_at = k;
        }
        CHECK(tracker.tracks().size() == 1);
        if (confirm_travel <= 0.0) {
            CHECK(moving_at == 1);
        } else {
            CHECK(moving_at * dt > 0.3 && moving_at * dt < 1.0);
        }
        const nav::Track& track = tracker.tracks()[0];
        CHECK_NEAR(track.velocity.x, -0.4, 0.06);
        CHECK_NEAR(track.velocity.y, 0.0, 0.03);
    }

    // 3. Карта: пока объект считается подвижным, она его не запоминает; с отметкой
    //    Settled он становится её частью и «новым» больше не выглядит.
    {
        const Pose2 sensor{0.012, 0.017, 0.0};  // сдвиг уводит точки с границ клеток
        nav::OccupancyGrid map(nav::MapConfig{}, Vec2{0.0, 0.0});
        std::vector<Vec2> dirs(360);
        for (int i = 0; i < 360; ++i) {
            const double a = scan.angle_min + i * scan.angle_increment;
            dirs[static_cast<std::size_t>(i)] = {std::cos(a), std::sin(a)};
        }
        const std::vector<float> open(360, 3.0F);
        for (int k = 0; k < 10; ++k) map.integrate_scan(sensor, dirs, open, scan.range_max);  // пол подтверждён
        make_scan(1.52, ranges, transient);
        std::vector<std::uint8_t> found;
        map.find_transient_hits(sensor, dirs, ranges, scan.range_max, found);
        CHECK(found[180] == 1 && found[90] == 0);
        int cx = 0, cy = 0;
        CHECK(map.cell_of(sensor.position() + Vec2{1.52, 0.0}, cx, cy));

        std::vector<std::uint8_t> ray_flags(360, nav::ray_flag::Static);
        for (std::size_t i = 178; i <= 182; ++i) ray_flags[i] = nav::ray_flag::Dynamic;
        for (int k = 0; k < 5; ++k) map.integrate_scan(sensor, dirs, ranges, scan.range_max, ray_flags);
        map.find_transient_hits(sensor, dirs, ranges, scan.range_max, found);
        CHECK(found[180] == 1);
        CHECK(map.state(cx, cy) == nav::CellState::Free);

        for (std::size_t i = 178; i <= 182; ++i) ray_flags[i] = nav::ray_flag::Settled;
        for (int k = 0; k < 5; ++k) map.integrate_scan(sensor, dirs, ranges, scan.range_max, ray_flags);
        map.find_transient_hits(sensor, dirs, ranges, scan.range_max, found);
        CHECK(found[180] == 0);
        CHECK(map.state(cx, cy) == nav::CellState::Occupied);
    }
}

TEST(occupancy_grid_remembers_hits_against_misses) {
    // Клетка, много раз наблюдавшаяся занятой, переживает столько же наблюдений
    // «свободно» (так выглядят лучи, прошедшие «сквозь» стену из-за ошибки позы).
    // Без памяти попаданий она стирается за десяток сканов. Забывание при этом
    // остаётся: когда запомненные попадания исчерпаны, клетка освобождается.
    const Pose2 sensor{0.012, 0.017, 0.0};
    std::vector<Vec2> dirs(360);
    for (int i = 0; i < 360; ++i) {
        const double a = -kPi + i * kTwoPi / 360;
        dirs[static_cast<std::size_t>(i)] = {std::cos(a), std::sin(a)};
    }
    const std::vector<float> open(360, 3.0F);
    std::vector<float> blocked = open;
    for (std::size_t i = 178; i <= 182; ++i) blocked[i] = 1.52F;  // предмет в 1.52 м прямо по курсу
    for (const int memory : {0, 200}) {
        nav::MapConfig config;
        config.hit_memory = memory;
        nav::OccupancyGrid map(config, Vec2{0.0, 0.0});
        int cx = 0, cy = 0;
        CHECK(map.cell_of(sensor.position() + Vec2{1.52, 0.0}, cx, cy));
        for (int k = 0; k < 60; ++k) map.integrate_scan(sensor, dirs, blocked, 10.0);
        CHECK(map.state(cx, cy) == nav::CellState::Occupied);
        for (int k = 0; k < 40; ++k) map.integrate_scan(sensor, dirs, open, 10.0);
        CHECK(map.state(cx, cy) == (memory > 0 ? nav::CellState::Occupied : nav::CellState::Free));
        for (int k = 0; k < 40; ++k) map.integrate_scan(sensor, dirs, open, 10.0);
        CHECK(map.state(cx, cy) == nav::CellState::Free);
    }
}

TEST(navigator_rejects_goal_outside_map) {
    // Карта конечна; цель за её пределами — ошибка настройки, о которой нужно
    // сообщить сразу, а не ждать маршрута, которого не будет.
    const SimConfig config;
    nav::ScanGeometry scan;
    scan.num_rays = 360;
    scan.angle_min = -kPi;
    scan.angle_increment = kTwoPi / 360;
    scan.range_max = 10.0;
    nav::Navigator navigator(nav::navigator_config(config.robot, config.control_dt()), scan, Pose2{1.0, 2.0, 0.0});
    navigator.set_goal({9.0, -6.0});  // в пределах карты 20 × 20 м с центром в точке старта
    CHECK(navigator.status() == nav::NavStatus::Driving);
    bool rejected = false;
    try {
        navigator.set_goal({12.5, 2.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

TEST(navigator_reaches_goals_in_unknown_maze) {
    // Полный контур: только лидар и одометрия с шумом, карта заранее не известна.
    SimConfig config;
    config.task.max_steps = 3000;
    config.task.terminate_on_collision = false;
    int reached = 0, collisions = 0;
    double efficiency = 0.0;
    const int episodes = 8;
    for (int episode = 1; episode <= episodes; ++episode) {
        const auto world = std::make_shared<const World>(World::maze(5, 5, 1.2, 0.2, 0.05, 200 + static_cast<std::uint64_t>(episode), 0.15));
        Simulator sim(config, world);
        sim.reset(static_cast<std::uint64_t>(episode));
        nav::Navigator navigator(nav::navigator_config(config.robot, config.control_dt()), nav::scan_geometry(sim.lidar_model()), sim.odom_pose());
        navigator.set_goal(sim.goal());
        StepResult r;
        double travelled = 0.0;
        Vec2 previous = sim.robot_state().pose.position();
        while (!r.terminated && !r.truncated) {
            const auto p = sim.proprio();
            r = sim.step(navigator.update(sim.lidar(), sim.odom_pose(), p[obs::OdomV], p[obs::OdomW]));
            collisions += r.collision ? 1 : 0;
            travelled += distance(previous, sim.robot_state().pose.position());
            previous = sim.robot_state().pose.position();
        }
        reached += r.success ? 1 : 0;
        if (r.success) efficiency += sim.shortest_path_length() / std::max(sim.shortest_path_length(), travelled);
        CHECK(distance(navigator.pose().position(), sim.robot_state().pose.position()) < 0.12);  // SLAM не потерялся
    }
    CHECK(reached == episodes);
    CHECK(collisions == 0);
    CHECK(efficiency / episodes > 0.8);  // SPL: в неизвестной карте путь близок к оптимальному
}

TEST(navigator_avoids_moving_obstacles) {
    // Арена с неподвижными и пятью подвижными препятствиями.
    SimConfig config;
    config.task.max_steps = 3000;
    config.task.terminate_on_collision = false;
    config.obstacles.count = 5;
    int reached = 0, clean = 0;
    const int episodes = 10;
    for (int episode = 1; episode <= episodes; ++episode) {
        const auto world = std::make_shared<const World>(World::cluttered(9.0, 9.0, 0.05, 5, 5, 100 + static_cast<std::uint64_t>(episode)));
        Simulator sim(config, world);
        sim.reset(static_cast<std::uint64_t>(episode));
        nav::Navigator navigator(nav::navigator_config(config.robot, config.control_dt()), nav::scan_geometry(sim.lidar_model()), sim.odom_pose());
        navigator.set_goal(sim.goal());
        StepResult r;
        int contacts = 0;
        while (!r.terminated && !r.truncated) {
            const auto p = sim.proprio();
            r = sim.step(navigator.update(sim.lidar(), sim.odom_pose(), p[obs::OdomV], p[obs::OdomW]));
            contacts += r.collision ? 1 : 0;
        }
        reached += r.success ? 1 : 0;
        clean += r.success && contacts == 0 ? 1 : 0;
    }
    CHECK(reached == episodes);
    CHECK(clean >= episodes - 2);  // редкое касание возможно, когда препятствие само въезжает в стоящего робота
}

}  // namespace

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0, failed = 0;
    for (const TestCase& test : registry()) {
        if (std::string_view(test.name).find(filter) == std::string_view::npos) continue;
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", test.name);
        std::fflush(stdout);
        try {
            test.run();
        } catch (const std::exception& e) {
            ++g_failures;
            std::printf("    FAILED: unexpected exception: %s\n", e.what());
        }
        const bool ok = g_failures == before;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", test.name);
        ++ran;
        failed += ok ? 0 : 1;
    }
    std::printf("\n%d tests, %d failed\n", ran, failed);
    return failed == 0 ? 0 : 1;
}
