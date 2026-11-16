// roversim_nav_bench — воспроизводимая оценка навигационного стека (SLAM + Theta* + DWA).
//
// Робот не знает ни карты, ни своей истинной позы: навигатор получает только скан
// лидара и одометрию. Истинное состояние симулятора используется лишь для
// подсчёта метрик. Программа печатает три блока:
//   1. static  — неизвестные лабиринты, варианты стека: успех, касания, SPL,
//                ошибка оценки положения; для тех же прогонов — качество карты и
//                ложные срабатывания трекера;
//   2. dynamic — миры с подвижными препятствиями: разбор каждого касания (кто
//                сближался, видел ли робот препятствие, сопровождал ли его) и
//                точность оценки скорости препятствий;
//   3. timing  — время такта навигатора и его структура по компонентам.
// Прогоны детерминированы: повторный запуск даёт те же числа (кроме времени).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "roversim/nav/navigator.hpp"
#include "roversim/simulator.hpp"

namespace {

using namespace roversim;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Options {
    int episodes = 50;
    std::string only = "all";
    // Переопределения параметров навигатора — для сравнительных прогонов.
    std::optional<double> confirm_travel;
    std::optional<double> startup_time;
};

Options g_options;

void print_usage() {
    std::puts(
        "Usage: roversim_nav_bench [options]\n"
        "\n"
        "  --episodes N          episodes per table row                      (default 50)\n"
        "  --only PART           static | dynamic | timing | all             (default all)\n"
        "  --confirm-travel X    tracker: displacement after which an object counts as moving, m\n"
        "                        (default 0: moving as soon as it is confirmed)\n"
        "  --startup-time X      navigator: observation pause after start, s (default 0)\n"
        "  -h, --help            show this help");
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--episodes") {
            opt.episodes = std::stoi(value());
        } else if (arg == "--only") {
            opt.only = value();
        } else if (arg == "--confirm-travel") {
            opt.confirm_travel = std::stod(value());
        } else if (arg == "--startup-time") {
            opt.startup_time = std::stod(value());
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + arg + " (see --help)");
        }
    }
    if (opt.episodes < 1) throw std::invalid_argument("--episodes must be positive");
    if (opt.only != "all" && opt.only != "static" && opt.only != "dynamic" && opt.only != "timing") {
        throw std::invalid_argument("--only must be static, dynamic, timing or all");
    }
    return opt;
}

// ---------------------------------------------------------------------------
// Сценарии
// ---------------------------------------------------------------------------

enum class WorldKind { Maze, WideMaze, Clutter };

/// Мир эпизода: у каждого эпизода свой seed генератора, поэтому карта всегда новая.
std::shared_ptr<const World> make_world(WorldKind kind, int episode) {
    const auto seed = static_cast<std::uint64_t>(100 + episode);
    switch (kind) {
        case WorldKind::Maze:  // 5 × 5 ячеек, проходы 1.2 м, 15 % пробитых стен
            return std::make_shared<const World>(World::maze(5, 5, 1.2, 0.2, 0.05, seed, 0.15));
        case WorldKind::WideMaze:  // 4 × 4 ячейки, проходы 1.8 м
            return std::make_shared<const World>(World::maze(4, 4, 1.8, 0.2, 0.05, seed, 0.3));
        case WorldKind::Clutter:  // арена 9 × 9 м с неподвижными препятствиями
            return std::make_shared<const World>(World::cluttered(9.0, 9.0, 0.05, 5, 5, seed));
    }
    throw std::logic_error("unknown world kind");
}

/// Настройки эпизода: касание его не обрывает (чтобы посчитать все касания),
/// лимит шагов щедрый — 200 с модельного времени.
SimConfig sim_config(int obstacles) {
    SimConfig config;
    config.task.max_steps = 4000;
    config.task.terminate_on_collision = false;
    config.obstacles.count = obstacles;
    return config;
}

/// Настройки навигатора по умолчанию с учётом переопределений из командной строки.
nav::NavigatorConfig default_navigator() {
    const SimConfig config;
    nav::NavigatorConfig nav_config = nav::navigator_config(config.robot, config.control_dt());
    if (g_options.confirm_travel) nav_config.tracker.confirm_travel = *g_options.confirm_travel;
    if (g_options.startup_time) nav_config.startup_time = *g_options.startup_time;
    return nav_config;
}

// ---------------------------------------------------------------------------
// Один эпизод с подсчётом метрик
// ---------------------------------------------------------------------------

struct EpisodeStats {
    bool success = false;
    std::uint64_t steps = 0;
    std::size_t replans = 0;
    int contact_steps = 0;    ///< шагов, на которых робот чего-либо касался
    int obstacle_events = 0;  ///< касаний подвижных препятствий (серия шагов подряд — одно событие)
    int wall_events = 0;      ///< касаний стен
    int robot_closing = 0;    ///< события, в которых робот сам сближался быстрее 0.1 м/с
    int tracked = 0;          ///< события, в которых препятствие сопровождалось как подвижное
    int visible = 0;          ///< события, перед которыми препятствие было видно дольше 1 с
    int cold_start = 0;       ///< события в первые 3 с эпизода, пока карта вокруг ещё не построена
    double robot_speed = 0;   ///< сумма модулей скорости робота в моменты касаний [м/с]
    double travelled = 0;     ///< пройденный путь [м]
    double shortest = 0;      ///< длина кратчайшего пути от старта до цели [м]
    double slam_error = 0;    ///< ошибка оценки положения в конце эпизода [м]
    double odom_error = 0;    ///< то же для чистой одометрии [м]
    double seconds = 0;       ///< время счёта эпизода: навигатор и симулятор [с]
    // Восприятие.
    long false_track_steps = 0;  ///< шагов с «подвижным» треком, которому не соответствует препятствие
    long track_samples = 0;      ///< сравнений «трек — истинное препятствие»
    double velocity_error = 0;   ///< сумма ошибок оценки скорости по этим сравнениям [м/с]
    long wall_cells = 0;         ///< клеток поверхности стен, которые робот наблюдал
    long wall_occupied = 0;      ///< из них на карте заняты
    long wall_free = 0;          ///< из них на карте свободны (ошибка картографирования)
};

/// Качество карты: в каком состоянии на карте робота находятся истинные клетки
/// поверхности стен, которые он наблюдал. Клетка считается наблюдавшейся, если
/// её или соседние клетки карты затронул хотя бы один луч.
void measure_map(const World& world, const nav::OccupancyGrid& map, EpisodeStats& stats) {
    for (int iy = 1; iy + 1 < world.height(); ++iy) {
        for (int ix = 1; ix + 1 < world.width(); ++ix) {
            if (!world.occupied(ix, iy)) continue;
            const bool surface = !world.occupied(ix + 1, iy) || !world.occupied(ix - 1, iy) ||
                                 !world.occupied(ix, iy + 1) || !world.occupied(ix, iy - 1);
            if (!surface) continue;
            int mx = 0, my = 0;
            if (!map.cell_of(world.cell_center(ix, iy), mx, my)) continue;
            if (mx < 1 || my < 1 || mx + 1 >= map.width() || my + 1 >= map.height()) continue;
            bool seen = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) seen |= map.log_odds(mx + dx, my + dy) != 0.0F;
            }
            if (!seen) continue;
            ++stats.wall_cells;
            const nav::CellState state = map.state(mx, my);
            stats.wall_occupied += state == nav::CellState::Occupied;
            stats.wall_free += state == nav::CellState::Free;
        }
    }
}

EpisodeStats run_episode(const SimConfig& config, const std::shared_ptr<const World>& world, std::uint64_t seed,
                         const nav::NavigatorConfig& nav_config) {
    Simulator sim(config, world);
    sim.reset(seed);
    nav::Navigator navigator(nav_config, nav::scan_geometry(sim.lidar_model()), sim.odom_pose());
    navigator.set_goal(sim.goal());

    EpisodeStats stats;
    stats.shortest = sim.shortest_path_length();
    std::vector<double> visible_time(sim.obstacles().size(), 0.0);
    Vec2 previous = sim.robot_state().pose.position();
    bool in_contact = false;
    StepResult result;
    const auto start = Clock::now();
    while (!result.terminated && !result.truncated) {
        // Состояние до шага: ближайшее препятствие и то, как долго каждое из них
        // находится в прямой видимости (по истинной карте — это метрика, не вход).
        const Pose2 pose = sim.robot_state().pose;
        const double robot_speed = sim.robot_state().velocity.x;
        int nearest = -1;
        double nearest_distance = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < sim.obstacles().size(); ++i) {
            const Vec2 offset = sim.obstacles()[i].position - pose.position();
            const double dist = norm(offset);
            const bool line_of_sight =
                dist < 1e-9 || sim.world().raycast(pose.position(), offset * (1.0 / dist), dist) >= dist - 1e-6;
            visible_time[i] = line_of_sight ? visible_time[i] + config.control_dt() : 0.0;
            if (dist < nearest_distance) {
                nearest_distance = dist;
                nearest = static_cast<int>(i);
            }
        }

        const std::span<const float> proprio = sim.proprio();
        const WheelCommand command =
            navigator.update(sim.lidar(), sim.odom_pose(), proprio[obs::OdomV], proprio[obs::OdomW]);

        // Треки подвижных объектов против истины (препятствия ещё не сдвинулись:
        // состояние симулятора соответствует скану, который видел навигатор).
        bool false_track = false;
        for (const nav::Track& track : navigator.tracks()) {
            if (!track.moving) continue;
            bool matched = false;
            for (const DynamicObstacle& obstacle : sim.obstacles()) {
                if (distance(track.position, obstacle.position) >= 0.35) continue;
                stats.velocity_error += distance(track.velocity, obstacle.velocity);
                ++stats.track_samples;
                matched = true;
                break;
            }
            false_track |= !matched;
        }
        stats.false_track_steps += false_track;

        result = sim.step(command);
        ++stats.steps;
        stats.contact_steps += result.collision;

        if (result.collision && !in_contact) {
            if (nearest >= 0 && nearest_distance < 0.75) {
                // Касание подвижного препятствия. Вклад робота в сближение — проекция
                // его скорости на направление к препятствию.
                const DynamicObstacle& obstacle = sim.obstacles()[static_cast<std::size_t>(nearest)];
                const Vec2 offset = obstacle.position - pose.position();
                const double dist = std::max(norm(offset), 1e-9);
                const double closing =
                    robot_speed * dot(Vec2{std::cos(pose.theta), std::sin(pose.theta)}, offset * (1.0 / dist));
                bool tracked = false;
                for (const nav::Track& track : navigator.tracks()) {
                    tracked |= track.moving && distance(track.position, obstacle.position) < 0.35;
                }
                ++stats.obstacle_events;
                stats.robot_closing += closing > 0.10;
                stats.tracked += tracked;
                stats.visible += visible_time[static_cast<std::size_t>(nearest)] > 1.0;
                stats.cold_start += sim.time() < 3.0;
                stats.robot_speed += std::abs(robot_speed);
            } else {
                ++stats.wall_events;
            }
        }
        in_contact = result.collision;

        const Vec2 position = sim.robot_state().pose.position();
        stats.travelled += distance(previous, position);
        previous = position;
    }
    stats.seconds = seconds_since(start);
    stats.success = result.success;
    stats.replans = navigator.replans();
    const Vec2 truth = sim.robot_state().pose.position();
    stats.slam_error = distance(navigator.pose().position(), truth);
    stats.odom_error = distance(sim.odom_pose().position(), truth);
    measure_map(*world, navigator.map(), stats);
    return stats;
}

/// Сумма метрик по эпизодам одной строки таблицы.
struct Totals {
    int episodes = 0, success = 0, clean = 0;
    int obstacle_events = 0, wall_events = 0, robot_closing = 0, tracked = 0, visible = 0, cold_start = 0;
    double robot_speed = 0, spl = 0, slam_error = 0, odom_error = 0, seconds = 0, velocity_error = 0;
    long false_track_steps = 0, track_samples = 0, wall_cells = 0, wall_occupied = 0, wall_free = 0;
    std::uint64_t steps = 0;
    std::size_t replans = 0;

    void add(const EpisodeStats& e) {
        ++episodes;
        success += e.success;
        clean += e.success && e.contact_steps == 0;
        obstacle_events += e.obstacle_events;
        wall_events += e.wall_events;
        robot_closing += e.robot_closing;
        tracked += e.tracked;
        visible += e.visible;
        cold_start += e.cold_start;
        robot_speed += e.robot_speed;
        // SPL: успех, взвешенный отношением кратчайшего пути к пройденному.
        if (e.success) spl += e.shortest / std::max(e.shortest, e.travelled);
        slam_error += e.slam_error;
        odom_error += e.odom_error;
        seconds += e.seconds;
        velocity_error += e.velocity_error;
        false_track_steps += e.false_track_steps;
        track_samples += e.track_samples;
        wall_cells += e.wall_cells;
        wall_occupied += e.wall_occupied;
        wall_free += e.wall_free;
        steps += e.steps;
        replans += e.replans;
    }
};

Totals run_row(WorldKind world, int obstacles, int episodes, const nav::NavigatorConfig& nav_config) {
    const SimConfig config = sim_config(obstacles);
    Totals totals;
    for (int episode = 1; episode <= episodes; ++episode) {
        totals.add(run_episode(config, make_world(world, episode), static_cast<std::uint64_t>(episode), nav_config));
    }
    return totals;
}

double percent(long part, long whole) { return whole > 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0; }

// ---------------------------------------------------------------------------
// 1. Неизвестные лабиринты без подвижных препятствий
// ---------------------------------------------------------------------------

void bench_static(int episodes) {
    struct Row {
        const char* name;
        bool scan_matching;
        nav::PlannerConfig::Algorithm algorithm;
        bool incidence_correction;
    };
    const Row rows[] = {
        {"SLAM + Theta* + DWA", true, nav::PlannerConfig::Algorithm::ThetaStar, true},
        {"A* + smoothing", true, nav::PlannerConfig::Algorithm::AStar, true},
        {"no scan matching", false, nav::PlannerConfig::Algorithm::ThetaStar, true},
        {"no incidence correction", true, nav::PlannerConfig::Algorithm::ThetaStar, false},
    };
    std::vector<Totals> totals;
    for (const Row& row : rows) {
        nav::NavigatorConfig nav_config = default_navigator();
        nav_config.slam.scan_matching = row.scan_matching;
        nav_config.planner.algorithm = row.algorithm;
        // Без поправки на угол падения свободным помечается весь луч до клетки отражения.
        if (!row.incidence_correction) nav_config.slam.map.surface_clearance = 0.0;
        totals.push_back(run_row(WorldKind::Maze, 0, episodes, nav_config));
    }

    std::printf("-- unknown mazes (5 x 5 cells, corridors 1.2 m), %d episodes per row --\n", episodes);
    std::puts("   configuration               success   contact-free    SPL   pose error   odometry error   replans/ep   steps/s");
    for (std::size_t i = 0; i < std::size(rows); ++i) {
        const Totals& t = totals[i];
        std::printf("   %-26s  %3d/%-3d      %3d/%-3d     %.3f    %5.1f cm        %5.1f cm        %5.1f    %6.0f\n",
                    rows[i].name, t.success, t.episodes, t.clean, t.episodes, t.spl / t.episodes,
                    100.0 * t.slam_error / t.episodes, 100.0 * t.odom_error / t.episodes,
                    static_cast<double>(t.replans) / t.episodes, static_cast<double>(t.steps) / t.seconds);
    }
    // Качество восприятия в тех же прогонах. Мир статичен, поэтому любой
    // «подвижный» трек — ложный.
    std::puts("   perception in the same runs:");
    std::puts("   configuration               wall cells seen   mapped occupied   mapped free (error)   steps with a false moving track");
    for (std::size_t i = 0; i < std::size(rows); ++i) {
        const Totals& t = totals[i];
        std::printf("   %-26s  %10ld         %5.1f %%           %5.2f %%                  %5.2f %%\n", rows[i].name, t.wall_cells,
                    percent(t.wall_occupied, t.wall_cells), percent(t.wall_free, t.wall_cells),
                    percent(t.false_track_steps, static_cast<long>(t.steps)));
    }
}

// ---------------------------------------------------------------------------
// 2. Подвижные препятствия
// ---------------------------------------------------------------------------

// Точность трекера в чистом виде: робот стоит в центре пустой арены 7 × 7 м, вокруг
// ездят два препятствия. Первые 3 с (накопление карты и захват) не учитываются.
void bench_tracker() {
    SimConfig config;
    config.obstacles.count = 2;
    config.task.max_steps = 0;
    config.task.terminate_on_collision = false;
    const auto arena = std::make_shared<const World>(World::arena(7.0, 7.0, 0.05));
    long checks = 0, matched = 0;
    double velocity_error = 0.0, position_error = 0.0;
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        Simulator sim(config, arena);
        sim.reset(seed, Pose2{3.5, 3.5, 0.0}, Vec2{1.0, 1.0});
        nav::Navigator navigator(default_navigator(), nav::scan_geometry(sim.lidar_model()), sim.odom_pose());
        for (int step = 0; step < 400; ++step) {
            navigator.update(sim.lidar(), sim.odom_pose(), 0.0, 0.0);
            if (step >= 60) {
                for (const DynamicObstacle& obstacle : sim.obstacles()) {
                    ++checks;
                    for (const nav::Track& track : navigator.tracks()) {
                        if (!track.moving || distance(track.position, obstacle.position) > 0.3) continue;
                        ++matched;
                        velocity_error += distance(track.velocity, obstacle.velocity);
                        position_error += distance(track.position, obstacle.position);
                        break;
                    }
                }
            }
            sim.step({0.0, 0.0});
        }
    }
    std::printf("   tracker alone (robot at rest in an empty 7 x 7 m arena, 2 obstacles, 6 runs of 20 s):\n"
                "          obstacle tracked %.0f %% of the time, velocity error %.3f m/s, position error %.3f m\n",
                percent(matched, checks), matched > 0 ? velocity_error / static_cast<double>(matched) : 0.0,
                matched > 0 ? position_error / static_cast<double>(matched) : 0.0);
}

void bench_dynamic(int episodes) {
    std::printf("-- moving obstacles (radius 0.2 m, speed 0.2-0.5 m/s), %d episodes per row --\n", episodes);
    std::puts("   world                      obstacles   success   contact-free   contacts   in first 3 s   robot closing   tracked   seen >1 s   robot speed   velocity error");
    struct Row {
        const char* name;
        WorldKind world;
        int obstacles;
    };
    const Row rows[] = {
        {"cluttered arena 9 x 9 m", WorldKind::Clutter, 5},
        {"maze, corridors 1.8 m", WorldKind::WideMaze, 4},
        {"maze, corridors 1.2 m", WorldKind::Maze, 2},
        {"maze, corridors 1.2 m", WorldKind::Maze, 4},
    };
    Totals all;
    const nav::NavigatorConfig nav_config = default_navigator();
    for (const Row& row : rows) {
        const Totals t = run_row(row.world, row.obstacles, episodes, nav_config);
        std::printf("   %-26s  %5d     %3d/%-3d      %3d/%-3d      %4d        %4d          %4d         %4d      %4d       %.2f m/s      %.3f m/s\n",
                    row.name, row.obstacles, t.success, t.episodes, t.clean, t.episodes, t.obstacle_events, t.cold_start,
                    t.robot_closing, t.tracked, t.visible,
                    t.obstacle_events > 0 ? t.robot_speed / t.obstacle_events : 0.0,
                    t.track_samples > 0 ? t.velocity_error / static_cast<double>(t.track_samples) : 0.0);
        all.obstacle_events += t.obstacle_events;
        all.wall_events += t.wall_events;
        all.robot_closing += t.robot_closing;
        all.tracked += t.tracked;
        all.visible += t.visible;
        all.cold_start += t.cold_start;
        all.robot_speed += t.robot_speed;
    }
    std::printf("   total: %d contacts with moving obstacles (%d in the first 3 s of an episode), %d with walls;\n"
                "          robot was closing in (> 0.1 m/s) in %d, obstacle was tracked as moving in %d,\n"
                "          in line of sight for > 1 s in %d; mean robot speed %.2f m/s\n",
                all.obstacle_events, all.cold_start, all.wall_events, all.robot_closing, all.tracked, all.visible,
                all.obstacle_events > 0 ? all.robot_speed / all.obstacle_events : 0.0);
    bench_tracker();
}

// ---------------------------------------------------------------------------
// 3. Время такта и его структура
// ---------------------------------------------------------------------------

// Навигатор не содержит хронометража, поэтому его этапы измеряются на «теневом»
// конвейере: отдельные экземпляры SLAM, трекера, планировщика и DWA получают в
// точности те же входы, что и настоящий навигатор (а значит, проходят те же
// состояния), и каждый вызов хронометрируется отдельно.
void bench_timing(int episodes) {
    const SimConfig config = sim_config(0);
    const nav::NavigatorConfig nc = default_navigator();
    nav::PlannerConfig astar_config = nc.planner;
    astar_config.algorithm = nav::PlannerConfig::Algorithm::AStar;
    double t_localize = 0, t_tracker = 0, t_integrate = 0, t_dwa = 0, t_costmap = 0, t_plan = 0, t_astar = 0;
    double t_navigator = 0, t_sim = 0, expansions = 0, astar_expansions = 0;
    long steps = 0, plans = 0, dwa_calls = 0;
    const auto micros = [](Clock::time_point start) { return 1e6 * seconds_since(start); };

    for (int episode = 1; episode <= episodes; ++episode) {
        Simulator sim(config, make_world(WorldKind::Maze, episode));
        sim.reset(static_cast<std::uint64_t>(episode));
        const nav::ScanGeometry scan = nav::scan_geometry(sim.lidar_model());
        nav::Navigator navigator(nc, scan, sim.odom_pose());
        navigator.set_goal(sim.goal());

        nav::GridSlam slam(nc.slam, scan, sim.odom_pose());
        nav::ObstacleTracker tracker(nc.tracker);
        nav::GridPlanner planner(nc.planner);
        nav::GridPlanner astar(astar_config);
        nav::DwaPlanner dwa(nc.dwa);
        std::vector<std::uint8_t> transient, ray_flags;
        std::vector<Vec2> points, path, ray_dirs_robot;
        std::vector<nav::MovingObstacle> moving;
        const double cm = std::cos(scan.mount.theta), sm = std::sin(scan.mount.theta);
        for (const Vec2& dir : slam.ray_directions()) ray_dirs_robot.push_back(rotate(dir, cm, sm));

        StepResult result;
        while (!result.terminated && !result.truncated) {
            const std::span<const float> proprio = sim.proprio();
            const std::span<const float> ranges = sim.lidar();
            const Pose2 odom = sim.odom_pose();

            // Восприятие: поза → подвижные объекты → карта.
            auto t = Clock::now();
            const Pose2 pose = slam.localize(odom, ranges);
            t_localize += micros(t);

            t = Clock::now();
            const Pose2 sensor = slam.sensor_pose();
            slam.map().find_transient_hits(sensor, slam.ray_directions(), ranges, scan.range_max, transient);
            tracker.update(sensor, scan, ranges, transient, nc.control_dt, ray_flags);
            t_tracker += micros(t);

            t = Clock::now();
            slam.integrate(ranges, ray_flags);
            t_integrate += micros(t);

            // Настоящий навигатор: полный такт.
            const std::size_t replans_before = navigator.replans();
            t = Clock::now();
            const WheelCommand command = navigator.update(ranges, odom, proprio[obs::OdomV], proprio[obs::OdomW]);
            t_navigator += micros(t);

            // Глобальный планировщик — на тех же тактах, что и у навигатора.
            if (navigator.replans() != replans_before) {
                t = Clock::now();
                planner.set_map(slam.map(), pose.position(), sim.goal());
                t_costmap += micros(t);
                t = Clock::now();
                planner.plan(pose.position(), sim.goal(), path);
                t_plan += micros(t);
                expansions += static_cast<double>(planner.expansions());
                // Для сравнения — A* со спрямлением на той же карте стоимости.
                astar.set_map(slam.map(), pose.position(), sim.goal());
                t = Clock::now();
                astar.plan(pose.position(), sim.goal(), path);
                t_astar += micros(t);
                astar_expansions += static_cast<double>(astar.expansions());
                ++plans;
            }

            // DWA — с теми же входами и на тех же тактах, что и у навигатора.
            const double c = std::cos(pose.theta), s = std::sin(pose.theta);
            const double reach = std::min(scan.range_max - 1e-3, nc.obstacle_range);
            points.clear();
            for (std::size_t i = 0; i < ranges.size(); ++i) {
                if (ranges[i] >= reach) continue;
                points.push_back(Vec2{scan.mount.x, scan.mount.y} + ray_dirs_robot[i] * static_cast<double>(ranges[i]));
            }
            moving.clear();
            for (const nav::Track& track : tracker.tracks()) {
                if (!track.moving) continue;
                moving.push_back({rotate_inv(track.position - pose.position(), c, s), rotate_inv(track.velocity, c, s),
                                  track.radius});
            }
            const Vec2 target = rotate_inv(navigator.carrot() - pose.position(), c, s);
            const double bearing = std::atan2(target.y, target.x);
            const bool threatened = std::any_of(moving.begin(), moving.end(), [&](const nav::MovingObstacle& m) {
                const double dist = norm(m.position);
                const double closing = dist > 1e-9 ? -dot(m.velocity, m.position) / dist : 0.0;
                return dist < m.radius + nc.dwa.robot_radius + 1.2 && closing > 0.05;
            });
            if (navigator.status() == nav::NavStatus::Driving &&
                (std::abs(bearing) <= nc.rotate_threshold || threatened)) {
                t = Clock::now();
                dwa.compute(points, moving, target, proprio[obs::OdomV], proprio[obs::OdomW], threatened);
                t_dwa += micros(t);
                ++dwa_calls;
            }

            t = Clock::now();
            result = sim.step(command);
            t_sim += micros(t);
            ++steps;
        }
    }

    const double n = static_cast<double>(steps);
    const double per_step = t_navigator / n;
    const double per_plan = plans > 0 ? 1.0 / static_cast<double>(plans) : 0.0;
    std::printf("-- navigator timing (unknown mazes, %d episodes, %ld control steps) --\n", episodes, steps);
    std::printf("   navigator step: %.0f us  (%.0f steps/s, %.0fx faster than the %.0f ms control period)\n", per_step,
                1e6 / per_step, 1e6 * config.control_dt() / per_step, 1e3 * config.control_dt());
    std::printf("   closed loop with the simulator: %.0f us per step  (%.0fx real time)\n", per_step + t_sim / n,
                1e6 * config.control_dt() / (per_step + t_sim / n));
    std::puts("   component                         per call      per step (amortized)");
    const auto line = [&](const char* name, double total, long calls) {
        std::printf("   %-30s  %8.1f us   %8.1f us\n", name, calls > 0 ? total / static_cast<double>(calls) : 0.0, total / n);
    };
    line("scan-to-map matching", t_localize, steps);
    line("map differencing + tracker", t_tracker, steps);
    line("scan integration", t_integrate, steps);
    line("DWA", t_dwa, dwa_calls);
    line("costmap", t_costmap, plans);
    line("Theta*", t_plan, plans);
    std::printf("   %-30s                %8.1f us\n", "sum of components",
                (t_localize + t_tracker + t_integrate + t_dwa + t_costmap + t_plan) / n);
    std::printf("   DWA runs on %.0f%% of steps; replanning every %.1f steps, %.0f expanded nodes per plan\n",
                100.0 * static_cast<double>(dwa_calls) / n, plans > 0 ? n / static_cast<double>(plans) : 0.0,
                expansions * per_plan);
    std::printf("   for comparison, A* + smoothing on the same costmaps: %.1f us per plan, %.0f expanded nodes\n",
                t_astar * per_plan, astar_expansions * per_plan);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        g_options = parse_options(argc, argv);
        const Options& opt = g_options;
        std::puts("roversim navigation benchmark: lidar 360 rays, sensor noise on, map and true pose unknown to the robot");
        if (opt.confirm_travel) std::printf("override: tracker.confirm_travel = %g m\n", *opt.confirm_travel);
        if (opt.startup_time) std::printf("override: startup_time = %g s\n", *opt.startup_time);
        if (opt.only == "all" || opt.only == "static") bench_static(opt.episodes);
        if (opt.only == "all" || opt.only == "dynamic") bench_dynamic(opt.episodes);
        if (opt.only == "all" || opt.only == "timing") bench_timing(opt.episodes);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "roversim_nav_bench: %s\n", e.what());
        return 1;
    }
}
