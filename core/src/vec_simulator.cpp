#include "roversim/vec_simulator.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace roversim {
namespace {

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

// Сколько итераций поток ждёт активно, прежде чем уснуть на condition_variable.
// Шаги идут один за другим, и активное ожидание (десятки микросекунд) убирает
// задержку пробуждения, сравнимую с длительностью самого шага.
constexpr int kSpinIterations = 4000;

}  // namespace

// Пул потоков с единственной операцией: разбить диапазон [0, count) на равные
// непрерывные куски и выполнить их параллельно. Вызывающий поток берёт первый кусок.
class VecSimulator::Pool {
 public:
    using Job = std::function<void(int begin, int end)>;

    explicit Pool(int threads) : threads_(std::max(1, threads)) {
        workers_.reserve(static_cast<std::size_t>(threads_ - 1));
        for (int id = 1; id < threads_; ++id) workers_.emplace_back([this, id] { worker(id); });
    }

    ~Pool() {
        {
            std::lock_guard lock(mutex_);
            stop_.store(true);
        }
        start_cv_.notify_all();
        for (std::thread& t : workers_) t.join();
    }

    void run(int count, const Job& job) {
        if (threads_ == 1 || count <= 1) {
            job(0, count);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            job_ = &job;
            count_ = count;
            error_ = nullptr;
            pending_.store(threads_ - 1);
            generation_.fetch_add(1);
        }
        start_cv_.notify_all();
        execute(0);
        for (int i = 0; i < kSpinIterations && pending_.load() != 0; ++i) cpu_relax();
        std::unique_lock lock(mutex_);
        done_cv_.wait(lock, [this] { return pending_.load() == 0; });
        if (error_) std::rethrow_exception(error_);
    }

 private:
    void execute(int id) noexcept {
        const int begin = static_cast<int>(static_cast<long long>(count_) * id / threads_);
        const int end = static_cast<int>(static_cast<long long>(count_) * (id + 1) / threads_);
        try {
            if (begin < end) (*job_)(begin, end);
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }

    void worker(int id) {
        std::uint64_t seen = 0;
        for (;;) {
            for (int i = 0; i < kSpinIterations; ++i) {
                if (generation_.load() != seen || stop_.load()) break;
                cpu_relax();
            }
            if (generation_.load() == seen && !stop_.load()) {
                std::unique_lock lock(mutex_);
                start_cv_.wait(lock, [&] { return generation_.load() != seen || stop_.load(); });
            }
            if (stop_.load()) return;
            seen = generation_.load();
            execute(id);
            if (pending_.fetch_sub(1) == 1) {
                // Блокировка гарантирует, что ожидающий поток не пропустит уведомление.
                std::lock_guard lock(mutex_);
                done_cv_.notify_one();
            }
        }
    }

    int threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_cv_, done_cv_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    const Job* job_ = nullptr;
    int count_ = 0;
    std::exception_ptr error_;
};

VecSimulator::VecSimulator(const SimConfig& config, std::vector<std::shared_ptr<const World>> worlds, int num_envs,
                           int num_threads, bool autoreset)
    : num_envs_(num_envs), num_rays_(config.lidar.num_rays), num_threads_(num_threads), autoreset_(autoreset) {
    if (num_envs < 1) throw std::invalid_argument("VecSimulator: num_envs must be >= 1");
    if (worlds.empty()) throw std::invalid_argument("VecSimulator: at least one world is required");
    if (num_rays_ < 1) throw std::invalid_argument("LidarConfig: num_rays must be >= 1");
    if (num_threads_ <= 0) num_threads_ = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    num_threads_ = std::min(num_threads_, num_envs_);

    const auto n = static_cast<std::size_t>(num_envs_);
    lidar_.assign(n * static_cast<std::size_t>(num_rays_), 0.0F);
    proprio_.assign(n * obs::Dim, 0.0F);
    state_.assign(n * gt::Dim, 0.0);
    rewards_.assign(n, 0.0F);
    terminated_.assign(n, 0);
    truncated_.assign(n, 0);
    collision_.assign(n, 0);
    success_.assign(n, 0);
    episode_index_.assign(n, 0);

    // Навигационная сетка зависит только от мира и радиуса робота, поэтому
    // строится один раз на мир и разделяется всеми его средами.
    const double nav_radius = config.robot.radius + config.task.spawn_clearance;
    std::map<const World*, std::shared_ptr<const NavGrid>> nav_cache;
    envs_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::shared_ptr<const World>& world = worlds[i % worlds.size()];
        if (!world) throw std::invalid_argument("VecSimulator: world is null");
        if (!world->finalized()) throw std::invalid_argument("VecSimulator: call World::finalize() first");
        std::shared_ptr<const NavGrid>& nav = nav_cache[world.get()];
        if (!nav) nav = std::make_shared<NavGrid>(*world, nav_radius);
        // Каждая среда пишет наблюдения прямо в свою строку общих массивов.
        const Simulator::Buffers buffers{lidar_.data() + i * static_cast<std::size_t>(num_rays_),
                                         proprio_.data() + i * obs::Dim, state_.data() + i * gt::Dim};
        envs_.emplace_back(config, world, buffers, nav);
    }
    pool_ = std::make_unique<Pool>(num_threads_);
    reset(0);
}

VecSimulator::~VecSimulator() = default;

std::uint64_t VecSimulator::episode_seed(int env, std::uint64_t episode) const noexcept {
    return mix64(mix64(base_seed_ + 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(env) + 1)) + episode);
}

void VecSimulator::reset(std::uint64_t seed) {
    base_seed_ = seed;
    pool_->run(num_envs_, [this](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const auto k = static_cast<std::size_t>(i);
            episode_index_[k] = 0;
            envs_[k].reset(episode_seed(i, 0));
            rewards_[k] = 0.0F;
            terminated_[k] = truncated_[k] = collision_[k] = success_[k] = 0;
        }
    });
}

void VecSimulator::step(std::span<const double> actions) {
    if (actions.size() != static_cast<std::size_t>(num_envs_) * 2) {
        throw std::invalid_argument("VecSimulator::step: actions must have shape [num_envs, 2]");
    }
    pool_->run(num_envs_, [this, actions](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const auto k = static_cast<std::size_t>(i);
            const StepResult r = envs_[k].step({actions[2 * k], actions[2 * k + 1]});
            rewards_[k] = static_cast<float>(r.reward);
            terminated_[k] = r.terminated;
            truncated_[k] = r.truncated;
            collision_[k] = r.collision;
            success_[k] = r.success;
            if (autoreset_ && (r.terminated || r.truncated)) {
                envs_[k].reset(episode_seed(i, ++episode_index_[k]));
            }
        }
    });
}

}  // namespace roversim
