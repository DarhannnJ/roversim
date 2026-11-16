// roversim_server — симулятор как отдельный процесс.
//
// Режим по умолчанию: сервер разделяемой памяти. Внешний скрипт (например,
// Python-клиент roversim.ShmClient) подключается к сегменту и управляет роботом.
//
// Режим --demo: клиент не нужен, робот в реальном времени сам ездит по
// случайным целям (а из веб-просмотрщика им можно управлять с клавиатуры).
//
// Визуализация включается флагом --viz; без него процесс работает headless.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <sys/ioctl.h>
#include <unistd.h>

#include "autopilot.hpp"
#include "roversim/nav/navigator.hpp"
#include "roversim/shm.hpp"
#include "roversim/viz.hpp"

namespace {

using namespace roversim;
using Clock = std::chrono::steady_clock;

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct Options {
    std::string shm = "/roversim";
    std::string world = "maze";
    std::uint64_t world_seed = 1;
    std::uint64_t seed = 1;
    int rays = 360;
    std::string viz = "none";
    int port = 8765;
    bool bind_all = false;
    bool demo = false;
    double speed = 1.0;
    int max_steps = -1;       // -1 — значение по умолчанию из TaskConfig
    int obstacles = 0;
    bool collision_stop = true;
    std::string driver = "pursuit";
    int spin_us = -1;         // -1 — значение по умолчанию (kShmDefaultSpinTime)
};

void print_usage() {
    std::puts(
        "Usage: roversim_server [options]\n"
        "\n"
        "  --shm NAME        shared memory segment name            (default /roversim)\n"
        "  --world KIND      maze | arena | clutter                (default maze)\n"
        "  --world-seed N    seed of the world generator           (default 1)\n"
        "  --rays N          number of lidar rays                  (default 360)\n"
        "  --max-steps N     episode step limit, 0 = unlimited     (default 600)\n"
        "  --obstacles N     number of moving obstacles            (default 0)\n"
        "  --no-collision-stop  a collision does not end the episode\n"
        "  --spin-us N       busy-wait for a command this long before sleeping, microseconds\n"
        "                    (default 200; raise it if the client thinks longer between steps\n"
        "                    and latency matters more than an idle CPU core)\n"
        "  --viz MODE        none | tui | web                      (default none: headless)\n"
        "  --port N          web viewer port                       (default 8765)\n"
        "  --bind-all        web viewer listens on all interfaces  (default: 127.0.0.1 only)\n"
        "  --demo            no client: drive autonomously in real time\n"
        "  --driver KIND     demo driver: pursuit (follows the true shortest path)\n"
        "                    or nav (SLAM + Theta* + DWA from lidar and odometry only)\n"
        "  --seed N          first episode seed in --demo mode     (default 1)\n"
        "  --speed X         real-time factor in --demo mode       (default 1.0)\n"
        "  -h, --help        show this help");
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--shm") {
            opt.shm = value();
        } else if (arg == "--world") {
            opt.world = value();
        } else if (arg == "--world-seed") {
            opt.world_seed = std::stoull(value());
        } else if (arg == "--seed") {
            opt.seed = std::stoull(value());
        } else if (arg == "--rays") {
            opt.rays = std::stoi(value());
        } else if (arg == "--max-steps") {
            opt.max_steps = std::stoi(value());
        } else if (arg == "--obstacles") {
            opt.obstacles = std::stoi(value());
        } else if (arg == "--no-collision-stop") {
            opt.collision_stop = false;
        } else if (arg == "--spin-us") {
            opt.spin_us = std::stoi(value());
            if (opt.spin_us < 0) throw std::invalid_argument("--spin-us must not be negative");
        } else if (arg == "--driver") {
            opt.driver = value();
        } else if (arg == "--viz") {
            opt.viz = value();
        } else if (arg == "--port") {
            opt.port = std::stoi(value());
        } else if (arg == "--speed") {
            opt.speed = std::stod(value());
        } else if (arg == "--bind-all") {
            opt.bind_all = true;
        } else if (arg == "--demo") {
            opt.demo = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + arg + " (see --help)");
        }
    }
    if (opt.viz != "none" && opt.viz != "tui" && opt.viz != "web") {
        throw std::invalid_argument("--viz must be none, tui or web");
    }
    if (opt.port < 0 || opt.port > 65535) throw std::invalid_argument("--port must be within 0..65535");
    if (!(opt.speed > 0.0)) throw std::invalid_argument("--speed must be positive");
    if (opt.driver != "pursuit" && opt.driver != "nav") throw std::invalid_argument("--driver must be pursuit or nav");
    return opt;
}

std::shared_ptr<const World> make_world(const Options& opt) {
    if (opt.world == "maze") return std::make_shared<World>(World::maze(5, 5, 1.2, 0.2, 0.05, opt.world_seed, 0.15));
    if (opt.world == "arena") return std::make_shared<World>(World::arena(8.0, 8.0, 0.05));
    if (opt.world == "clutter") return std::make_shared<World>(World::cluttered(8.0, 8.0, 0.05, 6, 6, opt.world_seed));
    throw std::invalid_argument("--world must be maze, arena or clutter");
}

/// Канал визуализации, выбранный флагом --viz.
class Display {
 public:
    explicit Display(const Options& opt) {
        if (opt.viz == "web") {
            web_ = std::make_unique<viz::WebViewer>(static_cast<std::uint16_t>(opt.port), opt.bind_all);
            std::printf("web viewer: http://%s:%u/\n", opt.bind_all ? "<host>" : "127.0.0.1", web_->port());
        } else if (opt.viz == "tui") {
            // Размер кадра — по размеру терминала: знакоместо вмещает 1×2 пикселя.
            winsize size{};
            int cols = 100, rows = 50;
            if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0 && size.ws_row > 0) {
                cols = size.ws_col;
                rows = size.ws_row;
            }
            tui_ = std::make_unique<viz::Renderer>(std::clamp(cols, 16, 400), std::clamp(2 * (rows - 2), 16, 400));
            std::fputs("\x1b[2J\x1b[?25l", stdout);  // очистить экран, скрыть курсор
        }
    }

    ~Display() {
        if (tui_) std::fputs("\x1b[0m\x1b[?25h\n", stdout);  // вернуть курсор
    }

    void show(const Simulator& sim, bool force = false) {
        if (web_) web_->publish(sim, force);
        if (tui_) {
            const auto now = Clock::now();
            if (!force && now - last_tui_ < std::chrono::milliseconds(66)) return;  // не чаще 15 кадров/с
            last_tui_ = now;
            const std::string frame = viz::to_ansi(tui_->render(sim));
            const RobotState& robot = sim.robot_state();
            std::fputs("\x1b[H", stdout);
            std::fwrite(frame.data(), 1, frame.size(), stdout);
            std::printf("\x1b[0mt=%7.2f s  episode %llu  v=%5.2f m/s  w=%5.2f rad/s  goal %5.2f m   \n",
                        sim.time(), static_cast<unsigned long long>(sim.episode()), robot.velocity.x,
                        robot.yaw_rate, distance(robot.pose.position(), sim.goal()));
            std::fflush(stdout);
        }
    }

    bool teleop(double& v, double& w) const { return web_ && web_->teleop(v, w); }

 private:
    std::unique_ptr<viz::WebViewer> web_;
    std::unique_ptr<viz::Renderer> tui_;
    Clock::time_point last_tui_{};
};

void run_server(const Options& opt, const SimConfig& config, std::shared_ptr<const World> world, Display& display) {
    ShmServer server(opt.shm, config, std::move(world));
    if (opt.spin_us >= 0) server.set_spin_time(1e-6 * opt.spin_us);
    std::printf("serving on shared memory %s (%d lidar rays, control period %.3f s)\n", server.name().c_str(),
                config.lidar.num_rays, config.control_dt());
    std::fflush(stdout);
    while (!g_stop.load()) {
        const ShmCommand command = server.serve_once(50);
        if (command == ShmCommand::Close) break;
        if (command != ShmCommand::None) display.show(server.simulator());
    }
}

void run_demo(const Options& opt, SimConfig config, std::shared_ptr<const World> world, Display& display) {
    // В демо столкновение не обрывает эпизод (иначе ручным управлением неудобно
    // пользоваться), а путь автопилота прокладывается с запасом от стен.
    config.task.terminate_on_collision = false;
    config.task.spawn_clearance = 0.18;
    if (opt.max_steps < 0) config.task.max_steps = 3000;
    Simulator sim(config, std::move(world));
    std::uint64_t seed = opt.seed;
    sim.reset(seed);

    // Водитель «nav» — полный автономный контур: он не видит ни истинной позы, ни
    // карты, только лидар и одометрию. Навигатор создаётся заново на каждый эпизод.
    const bool autonomous = opt.driver == "nav";
    std::unique_ptr<nav::Navigator> navigator;
    const auto start_navigator = [&] {
        if (!autonomous) return;
        navigator = std::make_unique<nav::Navigator>(nav::navigator_config(config.robot, config.control_dt()),
                                                     nav::scan_geometry(sim.lidar_model()), sim.odom_pose());
        navigator->set_goal(sim.goal());
    };
    start_navigator();

    const DiffDrive& drive = sim.drive();
    const auto period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(config.control_dt() / opt.speed));
    auto next = Clock::now();
    while (!g_stop.load()) {
        double v = 0.0, w = 0.0;
        // Навигатор обновляется каждый такт, даже при ручном управлении: его карта и
        // оценка позы должны следовать за роботом.
        WheelCommand command = pure_pursuit(sim);
        if (navigator) {
            const std::span<const float> p = sim.proprio();
            command = navigator->update(sim.lidar(), sim.odom_pose(), p[obs::OdomV], p[obs::OdomW]);
        }
        if (display.teleop(v, w)) {
            command = drive.twist_to_wheels(0.8 * v * drive.max_linear_speed(), 0.4 * w * drive.max_angular_speed());
        }
        const StepResult result = sim.step(command);
        const bool done = result.terminated || result.truncated;
        display.show(sim, done);
        if (done) {
            sim.reset(++seed);
            start_navigator();
        }

        // Привязка к реальному времени; после длинной паузы отставание не навёрстывается.
        next += period;
        const auto now = Clock::now();
        if (next < now) next = now;
        std::this_thread::sleep_until(next);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_options(argc, argv);
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        SimConfig config;
        config.lidar.num_rays = opt.rays;
        config.obstacles.count = opt.obstacles;
        config.task.terminate_on_collision = opt.collision_stop;
        if (opt.max_steps >= 0) config.task.max_steps = opt.max_steps;
        std::shared_ptr<const World> world = make_world(opt);
        Display display(opt);
        if (opt.demo) {
            run_demo(opt, config, std::move(world), display);
        } else {
            run_server(opt, config, std::move(world), display);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "roversim_server: %s\n", e.what());
        return 1;
    }
}
