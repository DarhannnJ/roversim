// Сеточный SLAM: одновременное построение карты занятости и оценка позы.
//
// Цикл на каждый скан:
//   1. Прогноз. Приращение одометрии между сканами переносит прошлую оценку позы.
//   2. Коррекция. Поза уточняется сопоставлением скана с уже построенной картой
//      (scan-to-map matching): ищется поза, при которой точки скана ложатся на
//      занятые клетки. Одометрия служит априорной оценкой и удерживает решение
//      там, где геометрия его не определяет (например, вдоль длинного коридора).
//   3. Картографирование. Скан встраивается в карту из уточнённой позы.
//
// Замыкания циклов нет: это «фронтенд» SLAM. Ошибка накапливается только при
// освоении новых областей; в уже снятой местности поза привязана к карте.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "roversim/math.hpp"
#include "roversim/nav/occupancy_grid.hpp"

namespace roversim::nav {

/// Геометрия скана лидара: всё, что нужно знать потребителю данных.
struct ScanGeometry {
    int num_rays = 360;
    double angle_min = -kPi;                 ///< угол первого луча в СК сенсора [рад]
    double angle_increment = kTwoPi / 360.0; ///< шаг между лучами [рад]
    double range_max = 10.0;                 ///< дальность «нет отражения» [м]
    Pose2 mount{};                           ///< поза сенсора в СК робота
};

struct SlamConfig {
    MapConfig map;
    bool scan_matching = true;   ///< false — картографирование по одной одометрии
    int warmup_scans = 5;        ///< столько первых сканов только строят карту
    int match_points = 120;      ///< сколько лучей скана участвует в сопоставлении
    double min_score = 0.25;     ///< результат с меньшей оценкой отбрасывается (мало общего с картой)
    double linear_step = 0.02;   ///< начальный шаг поиска по положению [м]
    double angular_step = 0.01;  ///< начальный шаг поиска по углу [рад]
    int refinements = 4;         ///< число уровней: на каждом шаг уменьшается вдвое
    int max_iterations = 10;     ///< шагов подъёма на одном уровне
    double prior_linear = 30.0;  ///< вес штрафа за отход от прогноза по положению [1/м²]
    double prior_angular = 60.0; ///< то же по углу [1/рад²]
};

class GridSlam {
 public:
    /// Карта центрируется в начальной позе; СК карты совпадает с СК, в которой
    /// задана initial_pose (обычно это начальная поза одометрии).
    GridSlam(const SlamConfig& config, const ScanGeometry& scan, const Pose2& initial_pose);

    /// Полный цикл: локализация и встраивание скана. Возвращает оценку позы.
    const Pose2& update(const Pose2& odom_pose, std::span<const float> ranges);

    /// Те же два шага по отдельности — когда между ними нужно решить, какие лучи
    /// попали в подвижные объекты и в карту идти не должны.
    const Pose2& localize(const Pose2& odom_pose, std::span<const float> ranges);
    void integrate(std::span<const float> ranges, std::span<const std::uint8_t> ray_flags = {});

    const Pose2& pose() const noexcept { return pose_; }
    Pose2 sensor_pose() const noexcept;
    const OccupancyGrid& map() const noexcept { return map_; }
    const ScanGeometry& scan() const noexcept { return scan_; }
    /// Единичные направления лучей в СК сенсора.
    std::span<const Vec2> ray_directions() const noexcept { return dirs_; }
    /// Оценка последнего сопоставления: средняя «занятость» карты под точками скана, [-1, 1].
    double match_score() const noexcept { return score_; }
    /// Была ли последняя оценка позы принята от сопоставления (а не от одометрии).
    bool matched() const noexcept { return matched_; }

 private:
    double map_score(const Pose2& robot) const noexcept;
    Pose2 match(const Pose2& predicted);

    SlamConfig cfg_;
    ScanGeometry scan_;
    OccupancyGrid map_;
    std::vector<Vec2> dirs_;     ///< направления лучей в СК сенсора
    std::vector<Vec2> points_;   ///< точки текущего скана в СК робота, отобранные для сопоставления
    Pose2 pose_{};
    Pose2 last_odom_{};
    bool has_odom_ = false;
    double score_ = 0.0;
    bool matched_ = false;
};

}  // namespace roversim::nav
