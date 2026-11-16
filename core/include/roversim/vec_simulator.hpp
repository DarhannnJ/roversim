// Пакетный симулятор: N независимых сред за один вызов step().
//
// Предназначен для массового обучения с подкреплением:
//   * наблюдения всех сред лежат в общих непрерывных массивах [N × dim], которые
//     без копирования превращаются в numpy-массив или torch-тензор;
//   * среды шагают параллельно в пуле потоков;
//   * завершившиеся эпизоды перезапускаются автоматически.
//
// Результат не зависит от числа потоков: среды не разделяют изменяемого
// состояния, а seed каждого эпизода определяется только базовым seed, номером
// среды и номером эпизода.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "roversim/simulator.hpp"

namespace roversim {

class VecSimulator {
 public:
    /// Среда i работает в мире worlds[i % worlds.size()]: можно дать один мир на
    /// всех или набор разных (например, разные лабиринты для обобщения).
    /// num_threads = 0 — по числу ядер. autoreset — перезапускать среду сразу
    /// после завершения эпизода.
    VecSimulator(const SimConfig& config, std::vector<std::shared_ptr<const World>> worlds, int num_envs,
                 int num_threads = 0, bool autoreset = true);
    ~VecSimulator();

    VecSimulator(const VecSimulator&) = delete;
    VecSimulator& operator=(const VecSimulator&) = delete;

    /// Перезапускает все среды. Эпизоды нумеруются заново.
    void reset(std::uint64_t seed);

    /// Шаг всех сред. actions — [N × 2]: уставки (левое, правое) колёс [рад/с].
    /// При autoreset у завершившейся среды в буферах наблюдений оказывается
    /// первое наблюдение нового эпизода, а rewards/terminated/truncated
    /// относятся к последнему шагу завершённого.
    void step(std::span<const double> actions);

    int num_envs() const noexcept { return num_envs_; }
    int num_rays() const noexcept { return num_rays_; }
    int num_threads() const noexcept { return num_threads_; }

    // ---- Пакетные буферы, построчно по средам ----
    std::span<const float> lidar() const noexcept { return lidar_; }      ///< [N × num_rays]
    std::span<const float> proprio() const noexcept { return proprio_; }  ///< [N × obs::Dim]
    std::span<const double> state() const noexcept { return state_; }     ///< [N × gt::Dim]
    std::span<const float> rewards() const noexcept { return rewards_; }  ///< [N]
    std::span<const std::uint8_t> terminated() const noexcept { return terminated_; }
    std::span<const std::uint8_t> truncated() const noexcept { return truncated_; }
    std::span<const std::uint8_t> collision() const noexcept { return collision_; }
    std::span<const std::uint8_t> success() const noexcept { return success_; }

    Simulator& env(int index) { return envs_.at(static_cast<std::size_t>(index)); }
    const Simulator& env(int index) const { return envs_.at(static_cast<std::size_t>(index)); }

 private:
    class Pool;

    std::uint64_t episode_seed(int env, std::uint64_t episode) const noexcept;

    int num_envs_;
    int num_rays_;
    int num_threads_;
    bool autoreset_;
    std::uint64_t base_seed_ = 0;

    std::vector<float> lidar_, proprio_, rewards_;
    std::vector<double> state_;
    std::vector<std::uint8_t> terminated_, truncated_, collision_, success_;
    std::vector<std::uint64_t> episode_index_;  ///< номер текущего эпизода каждой среды
    std::vector<Simulator> envs_;
    std::unique_ptr<Pool> pool_;
};

}  // namespace roversim
