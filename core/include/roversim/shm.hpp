// Межпроцессный интерфейс через разделяемую память POSIX.
//
// Симулятор работает в отдельном процессе (ShmServer), управляющий скрипт
// подключается к тому же сегменту (ShmClient). Обмен устроен как вызов
// «запрос — ответ» без единого копирования наблюдений:
//
//     клиент                               сервер
//     ------                               ------
//     пишет команду и действие
//     cmd_seq += 1, futex_wake    ──────►  просыпается, выполняет reset/step;
//                                          симулятор пишет наблюдения прямо
//                                          в сегмент
//     просыпается, читает         ◄──────  ack_seq = cmd_seq, futex_wake
//     наблюдения по указателю
//
// Обе стороны сначала ждут активно (спин — задержка порядка микросекунды), и
// только если ответа долго нет, засыпают на futex, чтобы не жечь процессор.
//
// Раскладка сегмента: [ShmHeader][state: f64 × gt::Dim][proprio: f32 × obs::Dim]
// [lidar: f32 × num_rays], каждый блок выровнен на 64 байта. Смещения записаны
// в заголовке, так что читать сегмент можно из любого языка.
//
// Один сегмент обслуживает одного управляющего клиента. Сторонние наблюдатели
// (логгер, визуализатор) подключаются в режиме «только чтение» и получают
// согласованные снимки через seqlock (ShmClient::snapshot).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "roversim/simulator.hpp"

namespace roversim {

inline constexpr std::uint32_t kShmMagic = 0x4D495352;  // "RSIM"
inline constexpr std::uint32_t kShmVersion = 1;
/// Сколько по умолчанию каждая сторона ждёт партнёра активно, прежде чем уснуть
/// на futex [с]. Шаг симуляции занимает десятки микросекунд, поэтому при
/// непрерывном обмене стороны дожидаются друг друга в спине и не платят за
/// системный вызов и пробуждение потока.
inline constexpr double kShmDefaultSpinTime = 200e-6;

enum class ShmCommand : std::uint32_t { None = 0, Reset = 1, Step = 2, Close = 3 };

namespace shm_flags {
enum : std::uint32_t {
    Terminated = 1U << 0,
    Truncated = 1U << 1,
    Collision = 1U << 2,
    Success = 1U << 3,
    Error = 1U << 4,    ///< команда отклонена (например, старт внутри препятствия)
    HasPose = 1U << 0,  ///< для cmd_flags: Reset с явными стартом и целью
};
}  // namespace shm_flags

/// Заголовок сегмента. Только типы фиксированной ширины; поля, которые пишут
/// разные процессы, разнесены по разным линиям кэша (нет false sharing).
struct ShmHeader {
    // ---- Описание (сервер пишет один раз; magic — последним, как признак готовности) ----
    std::atomic<std::uint32_t> magic;
    std::uint32_t version;
    std::uint64_t total_bytes;
    std::int64_t server_pid;
    std::uint32_t num_rays;
    std::uint32_t proprio_dim;
    std::uint32_t state_dim;
    std::uint32_t reserved0;
    std::uint64_t state_offset;    ///< смещения блоков от начала сегмента [байт]
    std::uint64_t proprio_offset;
    std::uint64_t lidar_offset;
    double control_dt;             ///< период управления [с]
    double lidar_angle_min;        ///< угол первого луча [рад]
    double lidar_angle_increment;  ///< шаг между лучами [рад]
    double lidar_range_min;
    double lidar_range_max;
    double lidar_mount[3];         ///< поза лидара в СК робота: x, y, theta
    double robot_radius;
    double wheel_radius;
    double track_width;
    double max_wheel_speed;

    // ---- Команда: пишет клиент ----
    alignas(64) std::atomic<std::uint32_t> cmd_seq;  ///< номер команды; futex-слово
    std::uint32_t cmd;                               ///< ShmCommand
    std::uint32_t cmd_flags;
    std::uint32_t reserved1;
    std::uint64_t seed;
    double action[2];  ///< уставки колёс (левое, правое) [рад/с]
    double start[3];   ///< x, y, theta — для Reset с HasPose
    double goal[2];

    // ---- Ответ: пишет сервер ----
    alignas(64) std::atomic<std::uint32_t> ack_seq;  ///< номер выполненной команды; futex-слово
    std::uint32_t status;                            ///< биты shm_flags
    double reward;
    double sim_time;
    std::uint64_t step_count;
    std::uint64_t episode;
    char error[96];  ///< текст ошибки при status & Error (строка с нулём на конце)

    // ---- Seqlock для сторонних читателей: нечётное значение — идёт запись ----
    alignas(64) std::atomic<std::uint32_t> state_seq;
};

static_assert(std::atomic<std::uint32_t>::is_always_lock_free, "atomics must be usable across processes");
static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t), "futex word must be 32 bits");
static_assert(alignof(ShmHeader) == 64);

class ShmServer {
 public:
    /// Создаёт сегмент `name` (например, "/roversim") и симулятор, пишущий
    /// наблюдения прямо в него. Сегмент, оставшийся от упавшего сервера,
    /// заменяется; занятый живым процессом — std::runtime_error.
    ShmServer(const std::string& name, const SimConfig& config, std::shared_ptr<const World> world);
    ~ShmServer();

    ShmServer(const ShmServer&) = delete;
    ShmServer& operator=(const ShmServer&) = delete;

    Simulator& simulator() noexcept { return *sim_; }
    const std::string& name() const noexcept { return name_; }

    /// Ждёт команду не дольше timeout_ms и выполняет её.
    /// Возвращает выполненную команду; None — команд не было.
    ShmCommand serve_once(int timeout_ms);

    /// Обслуживает клиента, пока stop == false и не получена команда Close.
    void serve(const std::atomic<bool>& stop);

    /// Сколько времени сервер выполнял последнюю команду [с]: от её получения до
    /// публикации ответа. Разность между временем запроса у клиента и этой
    /// величиной — чистая стоимость транспорта (см. roversim_bench).
    double last_command_seconds() const noexcept {
        return 1e-9 * static_cast<double>(last_command_ns_.load(std::memory_order_relaxed));
    }

    /// Время активного ожидания команды перед сном [с]. Если клиент между
    /// командами занят дольше этого времени (считает нейросеть), сервер засыпает,
    /// и его пробуждение добавляет к обмену десятки микросекунд. Увеличение
    /// времени убирает эту задержку ценой полностью занятого ядра процессора.
    /// Можно вызывать из другого потока, пока сервер работает.
    void set_spin_time(double seconds);
    double spin_time() const noexcept { return 1e-9 * static_cast<double>(spin_ns_.load(std::memory_order_relaxed)); }

 private:
    void publish_status(std::uint32_t extra_flags);

    std::string name_;
    void* base_ = nullptr;
    std::size_t size_ = 0;
    ShmHeader* header_ = nullptr;
    std::unique_ptr<Simulator> sim_;
    std::uint32_t last_seq_ = 0;
    std::atomic<std::uint64_t> last_command_ns_{0};
    std::atomic<std::int64_t> spin_ns_{static_cast<std::int64_t>(kShmDefaultSpinTime * 1e9)};
};

class ShmClient {
 public:
    /// Подключается к сегменту; ждёт готовности сервера не дольше timeout_s.
    /// read_only — режим наблюдателя: доступны только чтение буферов и snapshot().
    explicit ShmClient(const std::string& name, double timeout_s = 5.0, bool read_only = false);
    ~ShmClient();

    ShmClient(const ShmClient&) = delete;
    ShmClient& operator=(const ShmClient&) = delete;

    void reset(std::uint64_t seed);
    void reset(std::uint64_t seed, const Pose2& start, Vec2 goal);
    StepResult step(double left, double right);
    /// Просит сервер завершить обслуживание.
    void close_server();

    /// Время активного ожидания ответа перед сном [с] (см. ShmServer::set_spin_time).
    /// Имеет смысл увеличивать, если шаг симуляции длится дольше значения по умолчанию.
    void set_spin_time(double seconds);
    double spin_time() const noexcept { return 1e-9 * static_cast<double>(spin_ns_); }

    // ---- Наблюдения: указатели прямо в разделяемую память ----
    // Содержимое меняется при следующем reset()/step().
    std::span<const float> lidar() const noexcept { return {lidar_, header_->num_rays}; }
    std::span<const float> proprio() const noexcept { return {proprio_, header_->proprio_dim}; }
    std::span<const double> state() const noexcept { return {state_, header_->state_dim}; }

    const ShmHeader& header() const noexcept { return *header_; }
    double time() const noexcept { return header_->sim_time; }
    std::uint64_t step_count() const noexcept { return header_->step_count; }
    std::uint64_t episode() const noexcept { return header_->episode; }

    /// Согласованная копия наблюдений (seqlock): безопасна, даже когда сервер в
    /// это время шагает по командам другого клиента. false — не удалось за timeout_s.
    bool snapshot(std::span<float> lidar, std::span<float> proprio, std::span<double> state,
                  double timeout_s = 0.1) const;

 private:
    StepResult transact(ShmCommand command);

    std::string name_;
    void* base_ = nullptr;
    std::size_t size_ = 0;
    ShmHeader* header_ = nullptr;
    const float* lidar_ = nullptr;
    const float* proprio_ = nullptr;
    const double* state_ = nullptr;
    double timeout_s_;
    bool read_only_;
    std::int64_t spin_ns_ = static_cast<std::int64_t>(kShmDefaultSpinTime * 1e9);
};

}  // namespace roversim
