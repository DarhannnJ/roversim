#include "roversim/shm.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace roversim {
namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::nanoseconds;

constexpr std::size_t kCacheLine = 64;
constexpr std::size_t align_up(std::size_t value) noexcept {
    return (value + kCacheLine - 1) / kCacheLine * kCacheLine;
}

// Период, с которым ожидающий клиент проверяет, жив ли сервер.
constexpr auto kLivenessSlice = std::chrono::milliseconds(100);

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

#ifdef __linux__
// Futex без FUTEX_PRIVATE_FLAG: слово лежит в памяти, разделяемой между процессами.
void futex_wait(std::atomic<std::uint32_t>& word, std::uint32_t expected, nanoseconds timeout) noexcept {
    const timespec ts{static_cast<time_t>(timeout.count() / 1'000'000'000),
                      static_cast<long>(timeout.count() % 1'000'000'000)};
    ::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&word), FUTEX_WAIT, expected, &ts, nullptr, 0);
}
void futex_wake(std::atomic<std::uint32_t>& word) noexcept {
    ::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&word), FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}
#else
// Переносимая замена: короткий сон вместо futex (выше задержка, та же семантика).
void futex_wait(std::atomic<std::uint32_t>&, std::uint32_t, nanoseconds timeout) noexcept {
    std::this_thread::sleep_for(std::min<nanoseconds>(timeout, std::chrono::microseconds(50)));
}
void futex_wake(std::atomic<std::uint32_t>&) noexcept {}
#endif

/// Ждёт, пока слово не станет отличным от `seen`: сначала активно (не дольше
/// spin), затем во сне на futex. false — истёк timeout.
bool wait_change(std::atomic<std::uint32_t>& word, std::uint32_t seen, nanoseconds timeout, nanoseconds spin) noexcept {
    const auto start = Clock::now();
    const auto spin_until = start + std::min(spin, timeout);
    for (unsigned i = 0;; ++i) {
        if (word.load(std::memory_order_acquire) != seen) return true;
        cpu_relax();
        if ((i & 63U) == 63U && Clock::now() >= spin_until) break;
    }
    const auto deadline = start + timeout;
    for (;;) {
        if (word.load(std::memory_order_acquire) != seen) return true;
        const auto now = Clock::now();
        if (now >= deadline) return false;
        futex_wait(word, seen, deadline - now);
    }
}

/// Время активного ожидания в наносекундах; отрицательное и нечисловое — ошибка.
std::int64_t spin_nanoseconds(double seconds) {
    if (!(seconds >= 0.0) || seconds > 3600.0) throw std::invalid_argument("spin time must be within 0..3600 s");
    return static_cast<std::int64_t>(seconds * 1e9);
}

std::string normalize_name(const std::string& name) {
    if (name.empty()) throw std::invalid_argument("shared memory name is empty");
    std::string result = name.front() == '/' ? name : "/" + name;
    if (result.size() < 2 || result.find('/', 1) != std::string::npos) {
        throw std::invalid_argument("shared memory name must look like \"/name\"");
    }
    return result;
}

bool process_alive(std::int64_t pid) noexcept {
    if (pid <= 0) return false;
    return ::kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
}

[[noreturn]] void throw_errno(const std::string& what) {
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

/// Существующий сегмент обслуживается живым сервером?
/// Бросает исключение, если сегмент с таким именем принадлежит не roversim.
bool segment_in_use(const std::string& name) {
    const int fd = ::shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) return false;
    struct stat st {};
    bool in_use = false;
    if (::fstat(fd, &st) == 0 && static_cast<std::size_t>(st.st_size) >= sizeof(ShmHeader)) {
        void* base = ::mmap(nullptr, sizeof(ShmHeader), PROT_READ, MAP_SHARED, fd, 0);
        if (base != MAP_FAILED) {
            const auto* header = static_cast<const ShmHeader*>(base);
            const std::uint32_t magic = header->magic.load(std::memory_order_acquire);
            const std::int64_t pid = header->server_pid;
            ::munmap(base, sizeof(ShmHeader));
            if (magic != 0 && magic != kShmMagic) {
                ::close(fd);
                throw std::runtime_error("shared memory '" + name + "' exists and is not a roversim segment");
            }
            in_use = process_alive(pid);
        }
    }
    ::close(fd);
    return in_use;
}

}  // namespace

// ------------------------------------------------------------------ сервер

ShmServer::ShmServer(const std::string& name, const SimConfig& config, std::shared_ptr<const World> world)
    : name_(normalize_name(name)) {
    if (config.lidar.num_rays < 1) throw std::invalid_argument("LidarConfig: num_rays must be >= 1");
    const auto num_rays = static_cast<std::size_t>(config.lidar.num_rays);
    const std::size_t state_offset = align_up(sizeof(ShmHeader));
    const std::size_t proprio_offset = align_up(state_offset + gt::Dim * sizeof(double));
    const std::size_t lidar_offset = align_up(proprio_offset + obs::Dim * sizeof(float));
    size_ = align_up(lidar_offset + num_rays * sizeof(float));

    int fd = ::shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0 && errno == EEXIST) {
        if (segment_in_use(name_)) {
            throw std::runtime_error("shared memory '" + name_ + "' is already served by a running process");
        }
        ::shm_unlink(name_.c_str());  // остался от упавшего сервера
        fd = ::shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    }
    if (fd < 0) throw_errno("shm_open('" + name_ + "')");
    if (::ftruncate(fd, static_cast<off_t>(size_)) != 0) {
        ::close(fd);
        ::shm_unlink(name_.c_str());
        throw_errno("ftruncate");
    }
    base_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        ::shm_unlink(name_.c_str());
        throw_errno("mmap");
    }

    auto* bytes = static_cast<std::byte*>(base_);
    header_ = new (base_) ShmHeader{};  // память сегмента обнулена ядром; magic = 0 — «не готов»
    header_->server_pid = static_cast<std::int64_t>(::getpid());
    try {
        // Симулятор пишет наблюдения прямо в сегмент: между ним и клиентом нет ни одного memcpy.
        const Simulator::Buffers buffers{reinterpret_cast<float*>(bytes + lidar_offset),
                                         reinterpret_cast<float*>(bytes + proprio_offset),
                                         reinterpret_cast<double*>(bytes + state_offset)};
        sim_ = std::make_unique<Simulator>(config, std::move(world), buffers);
    } catch (...) {
        ::munmap(base_, size_);
        ::shm_unlink(name_.c_str());
        throw;
    }

    ShmHeader& h = *header_;
    h.version = kShmVersion;
    h.total_bytes = size_;
    h.num_rays = static_cast<std::uint32_t>(num_rays);
    h.proprio_dim = obs::Dim;
    h.state_dim = gt::Dim;
    h.state_offset = state_offset;
    h.proprio_offset = proprio_offset;
    h.lidar_offset = lidar_offset;
    h.control_dt = config.control_dt();
    h.lidar_angle_min = sim_->lidar_model().angle_min();
    h.lidar_angle_increment = sim_->lidar_model().angle_increment();
    h.lidar_range_min = config.lidar.range_min;
    h.lidar_range_max = config.lidar.range_max;
    h.lidar_mount[0] = config.lidar.mount.x;
    h.lidar_mount[1] = config.lidar.mount.y;
    h.lidar_mount[2] = config.lidar.mount.theta;
    h.robot_radius = config.robot.radius;
    h.wheel_radius = config.robot.wheel_radius;
    h.track_width = config.robot.track_width;
    h.max_wheel_speed = config.robot.max_wheel_speed;
    publish_status(0);
    h.magic.store(kShmMagic, std::memory_order_release);  // с этого момента клиенты могут подключаться
}

ShmServer::~ShmServer() {
    sim_.reset();
    if (header_ != nullptr) header_->magic.store(0, std::memory_order_release);  // сигнал клиентам: сервер ушёл
    if (base_ != nullptr) ::munmap(base_, size_);
    ::shm_unlink(name_.c_str());
}

void ShmServer::publish_status(std::uint32_t extra_flags) {
    const StepResult& r = sim_->last_result();
    std::uint32_t flags = extra_flags;
    if (r.terminated) flags |= shm_flags::Terminated;
    if (r.truncated) flags |= shm_flags::Truncated;
    if (r.collision) flags |= shm_flags::Collision;
    if (r.success) flags |= shm_flags::Success;
    ShmHeader& h = *header_;
    h.status = flags;
    h.reward = r.reward;
    h.sim_time = sim_->time();
    h.step_count = sim_->step_count();
    h.episode = sim_->episode();
}

ShmCommand ShmServer::serve_once(int timeout_ms) {
    ShmHeader& h = *header_;
    if (!wait_change(h.cmd_seq, last_seq_, std::chrono::milliseconds(std::max(0, timeout_ms)),
                     nanoseconds(spin_ns_.load(std::memory_order_relaxed)))) {
        return ShmCommand::None;
    }
    const auto started = Clock::now();
    const std::uint32_t seq = h.cmd_seq.load(std::memory_order_acquire);
    const auto command = static_cast<ShmCommand>(h.cmd);

    std::uint32_t extra_flags = 0;
    h.state_seq.fetch_add(1, std::memory_order_acq_rel);  // нечётное: идёт запись
    try {
        switch (command) {
            case ShmCommand::Reset:
                if ((h.cmd_flags & shm_flags::HasPose) != 0) {
                    sim_->reset(h.seed, Pose2{h.start[0], h.start[1], h.start[2]}, Vec2{h.goal[0], h.goal[1]});
                } else {
                    sim_->reset(h.seed);
                }
                break;
            case ShmCommand::Step:
                sim_->step({h.action[0], h.action[1]});
                break;
            default:
                break;
        }
    } catch (const std::exception& e) {
        // Некорректный запрос не должен ронять сервер: сообщаем клиенту и ждём дальше.
        extra_flags = shm_flags::Error;
        std::snprintf(h.error, sizeof(h.error), "%s", e.what());
    }
    publish_status(extra_flags);
    h.state_seq.fetch_add(1, std::memory_order_release);  // чётное: данные согласованы

    // Записывается до ответа: клиент, дождавшийся ack, видит время именно этой команды.
    last_command_ns_.store(
        static_cast<std::uint64_t>(std::chrono::duration_cast<nanoseconds>(Clock::now() - started).count()),
        std::memory_order_relaxed);
    last_seq_ = seq;
    h.ack_seq.store(seq, std::memory_order_release);
    futex_wake(h.ack_seq);
    return command;
}

void ShmServer::set_spin_time(double seconds) { spin_ns_.store(spin_nanoseconds(seconds), std::memory_order_relaxed); }

void ShmServer::serve(const std::atomic<bool>& stop) {
    while (!stop.load(std::memory_order_relaxed)) {
        if (serve_once(100) == ShmCommand::Close) break;
    }
}

// ------------------------------------------------------------------ клиент

ShmClient::ShmClient(const std::string& name, double timeout_s, bool read_only)
    : name_(normalize_name(name)), timeout_s_(timeout_s), read_only_(read_only) {
    const auto deadline = Clock::now() + std::chrono::duration_cast<nanoseconds>(std::chrono::duration<double>(timeout_s));
    // Сервер мог ещё не успеть создать или заполнить сегмент — пробуем до истечения срока.
    for (;;) {
        const int fd = ::shm_open(name_.c_str(), read_only ? O_RDONLY : O_RDWR, 0);
        if (fd >= 0) {
            struct stat st {};
            const bool sized = ::fstat(fd, &st) == 0 && static_cast<std::size_t>(st.st_size) >= sizeof(ShmHeader);
            void* base = sized ? ::mmap(nullptr, static_cast<std::size_t>(st.st_size),
                                        read_only ? PROT_READ : PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
                               : MAP_FAILED;
            ::close(fd);
            if (base != MAP_FAILED) {
                auto* header = static_cast<ShmHeader*>(base);
                if (header->magic.load(std::memory_order_acquire) == kShmMagic) {
                    base_ = base;
                    size_ = static_cast<std::size_t>(st.st_size);
                    header_ = header;
                    break;
                }
                ::munmap(base, static_cast<std::size_t>(st.st_size));
            }
        }
        if (Clock::now() >= deadline) {
            throw std::runtime_error("ShmClient: no roversim server on shared memory '" + name_ + "'");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    const ShmHeader& h = *header_;
    const bool layout_ok = h.version == kShmVersion && h.total_bytes <= size_ &&
                           h.state_offset + h.state_dim * sizeof(double) <= size_ &&
                           h.proprio_offset + h.proprio_dim * sizeof(float) <= size_ &&
                           h.lidar_offset + h.num_rays * sizeof(float) <= size_;
    if (!layout_ok) {
        ::munmap(base_, size_);
        throw std::runtime_error("ShmClient: incompatible segment layout (version mismatch?)");
    }
    const auto* bytes = static_cast<const std::byte*>(base_);
    state_ = reinterpret_cast<const double*>(bytes + h.state_offset);
    proprio_ = reinterpret_cast<const float*>(bytes + h.proprio_offset);
    lidar_ = reinterpret_cast<const float*>(bytes + h.lidar_offset);
}

ShmClient::~ShmClient() {
    if (base_ != nullptr) ::munmap(base_, size_);
}

void ShmClient::set_spin_time(double seconds) { spin_ns_ = spin_nanoseconds(seconds); }

StepResult ShmClient::transact(ShmCommand command) {
    if (read_only_) throw std::logic_error("ShmClient: a read-only client cannot send commands");
    ShmHeader& h = *header_;
    h.cmd = static_cast<std::uint32_t>(command);
    const std::uint32_t seq = h.cmd_seq.load(std::memory_order_relaxed) + 1;
    h.cmd_seq.store(seq, std::memory_order_release);
    futex_wake(h.cmd_seq);

    const auto deadline = Clock::now() + std::chrono::duration_cast<nanoseconds>(std::chrono::duration<double>(timeout_s_));
    for (;;) {
        const std::uint32_t ack = h.ack_seq.load(std::memory_order_acquire);
        if (ack == seq) break;
        if (wait_change(h.ack_seq, ack, kLivenessSlice, nanoseconds(spin_ns_))) continue;
        if (h.magic.load(std::memory_order_acquire) != kShmMagic) {
            throw std::runtime_error("ShmClient: the server has shut down");
        }
        if (Clock::now() >= deadline) {
            // PID сервера может быть не виден из другого контейнера, поэтому он
            // лишь уточняет сообщение, а не служит основанием для раннего отказа.
            throw std::runtime_error(process_alive(h.server_pid)
                                         ? "ShmClient: timed out waiting for the server"
                                         : "ShmClient: the server process is gone");
        }
    }

    if ((h.status & shm_flags::Error) != 0) {
        const std::size_t length = ::strnlen(h.error, sizeof(h.error));
        throw std::runtime_error("ShmClient: server rejected the command: " + std::string(h.error, length));
    }
    StepResult r;
    r.reward = h.reward;
    r.terminated = (h.status & shm_flags::Terminated) != 0;
    r.truncated = (h.status & shm_flags::Truncated) != 0;
    r.collision = (h.status & shm_flags::Collision) != 0;
    r.success = (h.status & shm_flags::Success) != 0;
    return r;
}

void ShmClient::reset(std::uint64_t seed) {
    if (read_only_) throw std::logic_error("ShmClient: a read-only client cannot send commands");
    header_->seed = seed;
    header_->cmd_flags = 0;
    transact(ShmCommand::Reset);
}

void ShmClient::reset(std::uint64_t seed, const Pose2& start, Vec2 goal) {
    if (read_only_) throw std::logic_error("ShmClient: a read-only client cannot send commands");
    ShmHeader& h = *header_;
    h.seed = seed;
    h.cmd_flags = shm_flags::HasPose;
    h.start[0] = start.x;
    h.start[1] = start.y;
    h.start[2] = start.theta;
    h.goal[0] = goal.x;
    h.goal[1] = goal.y;
    transact(ShmCommand::Reset);
}

StepResult ShmClient::step(double left, double right) {
    if (read_only_) throw std::logic_error("ShmClient: a read-only client cannot send commands");
    header_->action[0] = left;
    header_->action[1] = right;
    return transact(ShmCommand::Step);
}

void ShmClient::close_server() { transact(ShmCommand::Close); }

bool ShmClient::snapshot(std::span<float> lidar, std::span<float> proprio, std::span<double> state,
                         double timeout_s) const {
    const ShmHeader& h = *header_;
    const auto deadline = Clock::now() + std::chrono::duration_cast<nanoseconds>(std::chrono::duration<double>(timeout_s));
    for (;;) {
        // Seqlock: копия верна, если счётчик чётный и не изменился за время копирования.
        const std::uint32_t before = h.state_seq.load(std::memory_order_acquire);
        if ((before & 1U) == 0U) {
            std::memcpy(lidar.data(), lidar_, std::min<std::size_t>(lidar.size(), h.num_rays) * sizeof(float));
            std::memcpy(proprio.data(), proprio_, std::min<std::size_t>(proprio.size(), h.proprio_dim) * sizeof(float));
            std::memcpy(state.data(), state_, std::min<std::size_t>(state.size(), h.state_dim) * sizeof(double));
            std::atomic_thread_fence(std::memory_order_acquire);
            if (h.state_seq.load(std::memory_order_acquire) == before) return true;
        }
        if (Clock::now() >= deadline) return false;
        cpu_relax();
    }
}

}  // namespace roversim
