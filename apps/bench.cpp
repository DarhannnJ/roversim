// roversim_bench — замеры производительности ядра:
//   1. один симулятор при разном числе лучей лидара;
//   2. пакетный VecSimulator при разном числе потоков;
//   3. задержка обмена через разделяемую память (запрос — ответ).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "roversim/shm.hpp"
#include "roversim/vec_simulator.hpp"

namespace {

using namespace roversim;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Детерминированное «блуждающее» управление: робот ездит, поворачивает и иногда упирается в стены.
WheelCommand wander(std::uint64_t step, int env = 0) {
    const std::uint64_t phase = (step / 40 + static_cast<std::uint64_t>(env) * 7) % 5;
    constexpr double kLeft[5] = {14.0, 16.0, 6.0, 16.0, -6.0};
    constexpr double kRight[5] = {14.0, 6.0, 16.0, 16.0, 6.0};
    return {kLeft[phase], kRight[phase]};
}

SimConfig bench_config(int rays) {
    SimConfig config;
    config.lidar.num_rays = rays;
    config.task.terminate_on_collision = false;  // чтобы длина эпизодов не влияла на замер
    config.task.max_steps = 500;
    return config;
}

void bench_single(const std::shared_ptr<const World>& world) {
    std::puts("-- single simulator (maze 7.2 x 7.2 m, cell 5 cm, 10 physics substeps per step) --");
    std::puts("   rays     steps/s    us/step   real-time factor");
    for (const int rays : {90, 360, 1080}) {
        const SimConfig config = bench_config(rays);
        Simulator sim(config, world);
        sim.reset(1);
        const int steps = 40000;
        std::uint64_t seed = 1;
        double sink = 0.0;
        const auto start = Clock::now();
        for (int i = 0; i < steps; ++i) {
            const StepResult r = sim.step(wander(static_cast<std::uint64_t>(i)));
            sink += sim.lidar()[0];
            if (r.terminated || r.truncated) sim.reset(++seed);
        }
        const double elapsed = seconds_since(start);
        std::printf("   %4d  %10.0f  %9.2f  %10.0fx   (checksum %.3f)\n", rays, steps / elapsed,
                    1e6 * elapsed / steps, steps * config.control_dt() / elapsed, sink);
    }
}

void bench_vector(const std::shared_ptr<const World>& world) {
    const int envs = 256;
    std::printf("-- VecSimulator (%d envs, 360 rays) --\n", envs);
    std::puts("   threads     steps/s   speedup");
    const int max_threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    double base = 0.0;
    for (int threads = 1; threads <= max_threads; threads *= 2) {
        VecSimulator vec(bench_config(360), {world}, envs, threads);
        vec.reset(1);
        std::vector<double> actions(static_cast<std::size_t>(envs) * 2);
        const int iterations = 300;
        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            for (int e = 0; e < envs; ++e) {
                const WheelCommand c = wander(static_cast<std::uint64_t>(i), e);
                actions[2 * static_cast<std::size_t>(e)] = c.left;
                actions[2 * static_cast<std::size_t>(e) + 1] = c.right;
            }
            vec.step(actions);
        }
        const double rate = static_cast<double>(iterations) * envs / seconds_since(start);
        if (threads == 1) base = rate;
        std::printf("   %7d  %10.0f   %6.2fx\n", threads, rate, rate / base);
    }
}

void bench_shm(const std::shared_ptr<const World>& world) {
    std::puts("-- shared memory round trip (client step() -> server -> client), 360 rays --");
    const std::string name = "/roversim_bench_" + std::to_string(::getpid());
    const SimConfig config = bench_config(360);
    std::atomic<bool> stop{false};
    ShmServer server(name, config, world);
    std::thread server_thread([&] { server.serve(stop); });
    {
        ShmClient client(name);
        const auto percentile = [](const std::vector<double>& sorted, double p) {
            return sorted[static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1))];
        };
        // Для каждого шага — время запроса у клиента и время, которое сервер
        // потратил на сам шаг симуляции. Их разность — стоимость транспорта:
        // публикация команды, пробуждение сервера, публикация ответа,
        // пробуждение клиента. Измерение парное (обе величины относятся к одному
        // и тому же шагу), поэтому дрейф частоты процессора и разброс
        // длительности шага в разность не попадают.
        const auto measure = [&](const char* label, int steps, std::chrono::microseconds think_time) {
            client.reset(1);
            std::vector<double> round_trip_us(static_cast<std::size_t>(steps));
            std::vector<double> served_us(static_cast<std::size_t>(steps));
            std::vector<double> overhead_us(static_cast<std::size_t>(steps));
            std::uint64_t seed = 1;
            double busy = 0.0;
            for (int i = 0; i < steps; ++i) {
                if (think_time.count() > 0) std::this_thread::sleep_for(think_time);
                const WheelCommand c = wander(static_cast<std::uint64_t>(i));
                const auto t0 = Clock::now();
                const StepResult r = client.step(c.left, c.right);
                const auto k = static_cast<std::size_t>(i);
                round_trip_us[k] = 1e6 * seconds_since(t0);
                served_us[k] = 1e6 * server.last_command_seconds();
                overhead_us[k] = round_trip_us[k] - served_us[k];
                busy += round_trip_us[k];
                if (r.terminated || r.truncated) client.reset(++seed);
            }
            std::sort(round_trip_us.begin(), round_trip_us.end());
            std::sort(served_us.begin(), served_us.end());
            std::sort(overhead_us.begin(), overhead_us.end());
            std::printf("   %s\n", label);
            std::printf("      request latency: p50 %.1f us, p99 %.1f us, max %.1f us  (%.0f steps/s)\n",
                        percentile(round_trip_us, 0.50), percentile(round_trip_us, 0.99), round_trip_us.back(),
                        1e6 * steps / busy);
            std::printf("      simulation step inside the server: p50 %.1f us\n", percentile(served_us, 0.50));
            std::printf("      IPC overhead (request latency - server step time): p50 %.2f us, p99 %.2f us\n",
                        percentile(overhead_us, 0.50), percentile(overhead_us, 0.99));
        };
        // Непрерывный обмен: обе стороны дожидаются друг друга в активном ожидании.
        measure("back-to-back requests (both sides spin):", 20000, std::chrono::microseconds(0));
        // Клиент «думает» 1 мс между запросами (например, считает нейросеть):
        // сервер успевает уснуть на futex, и к обмену добавляется его пробуждение.
        measure("1 ms pause between requests (server sleeps on futex):", 2000, std::chrono::microseconds(1000));
        // То же, но сервер ждёт команду активно дольше паузы клиента и не засыпает.
        server.set_spin_time(5e-3);
        measure("1 ms pause between requests, server spin time 5 ms:", 2000, std::chrono::microseconds(1000));
        server.set_spin_time(kShmDefaultSpinTime);
        client.close_server();
    }
    server_thread.join();
}

}  // namespace

int main() {
    const auto world = std::make_shared<const World>(World::maze(5, 5, 1.2, 0.2, 0.05, 7, 0.15));
    std::printf("roversim benchmark: %u hardware threads\n", std::thread::hardware_concurrency());
    bench_single(world);
    bench_vector(world);
    bench_shm(world);
    return 0;
}
