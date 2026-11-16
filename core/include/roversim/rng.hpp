// Детерминированный генератор случайных чисел.
//
// Стандартные std::normal_distribution / std::uniform_real_distribution дают разные
// последовательности в libstdc++, libc++ и MSVC, поэтому для воспроизводимости
// генератор и все распределения реализованы здесь явно.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

namespace roversim {

/// Финализатор SplitMix64: биекция 64→64 бит с хорошим лавинным эффектом.
constexpr std::uint64_t mix64(std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// xoshiro256++ (Blackman, Vigna): 256 бит состояния, ~1 нс на число.
class Rng {
 public:
    explicit constexpr Rng(std::uint64_t seed = 0) noexcept { reseed(seed); }

    constexpr void reseed(std::uint64_t seed) noexcept {
        // Состояние разворачивается из seed через SplitMix64, как рекомендуют авторы.
        for (auto& s : s_) {
            seed += 0x9E3779B97F4A7C15ULL;
            s = mix64(seed);
        }
        has_spare_ = false;
    }

    /// Независимый поток с номером `stream` для того же master-seed.
    /// У каждого источника шума свой поток: изменение, например, числа лучей
    /// лидара не сдвигает последовательность шума IMU.
    static constexpr Rng stream(std::uint64_t seed, std::uint64_t stream) noexcept {
        return Rng(mix64(seed) ^ mix64(0xD1B54A32D192ED03ULL * (stream + 1)));
    }

    constexpr std::uint64_t next_u64() noexcept {
        const std::uint64_t result = std::rotl(s_[0] + s_[3], 23) + s_[0];
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);
        return result;
    }

    /// Равномерно на [0, 1): 53 старших бита → double без потери равномерности.
    constexpr double uniform() noexcept { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }
    constexpr double uniform(double lo, double hi) noexcept { return lo + (hi - lo) * uniform(); }

    /// Равномерное целое на [0, n), n > 0. Отбраковка по маске — без смещения.
    constexpr std::uint64_t below(std::uint64_t n) noexcept {
        const std::uint64_t mask = std::bit_ceil(n) - 1;
        std::uint64_t r = 0;
        do {
            r = next_u64() & mask;
        } while (r >= n);
        return r;
    }

    /// Стандартное нормальное N(0, 1), полярный метод Марсальи.
    /// Из трансцендентных функций нужен только log (sqrt и деление округляются
    /// точно), что сводит к минимуму зависимость от libm.
    double normal() noexcept {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u = 0.0, v = 0.0, s = 0.0;
        do {
            u = 2.0 * uniform() - 1.0;
            v = 2.0 * uniform() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double f = std::sqrt(-2.0 * std::log(s) / s);
        spare_ = v * f;
        has_spare_ = true;
        return u * f;
    }

 private:
    std::uint64_t s_[4]{};
    double spare_ = 0.0;
    bool has_spare_ = false;
};

}  // namespace roversim
