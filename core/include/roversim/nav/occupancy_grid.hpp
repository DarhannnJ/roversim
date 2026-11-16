// Карта занятости (occupancy grid) в лог-шансах.
//
// Каждая клетка хранит l = log(p / (1 - p)), где p — вероятность того, что
// клетка занята. Байесовское обновление по независимым наблюдениям в лог-шансах
// превращается в сложение:
//     l <- l + l_hit    для клетки, в которой луч лидара закончился;
//     l <- l - l_miss   для клеток, через которые луч прошёл.
// Значения ограничены сверху и снизу: карта не становится «самоуверенной» и
// способна забывать — клетка, откуда ушёл подвижный объект, снова освобождается.
//
// Клетки вдоль луча перечисляются целочисленным алгоритмом Брезенхема. Его
// накопитель ошибки инициализируется по фактическому положению концов луча
// внутри клеток (фиксированная точка, 1/256 клетки), а не по центрам клеток:
// иначе при скользящем падении линия заходила бы в ряд клеток стены раньше
// самого луча и стирала бы стену.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "roversim/math.hpp"

namespace roversim::nav {

struct MapConfig {
    double resolution = 0.05;        ///< сторона клетки [м]
    double size_x = 20.0;            ///< размеры карты [м]; центр карты — в точке старта
    double size_y = 20.0;
    double log_odds_hit = 0.85;      ///< прибавка для клетки с отражением (p ≈ 0.70)
    double log_odds_miss = 0.40;     ///< убыль для клетки на пути луча (p ≈ 0.40)
    double log_odds_min = -2.0;      ///< нижняя граница насыщения (p ≈ 0.12)
    double log_odds_max = 3.5;       ///< верхняя граница насыщения (p ≈ 0.97)
    double occupied_threshold = 0.6; ///< выше — клетка считается занятой (p > 0.65)
    double free_threshold = -0.6;    ///< ниже — свободной (p < 0.35)
    double free_range = 8.0;         ///< луч без отражения очищает клетки только до этой дальности [м]
    double surface_clearance = 1.5;  ///< клетки ближе этого расстояния к поверхности, в которую упёрся
                                     ///< луч, не помечаются свободными [клеток] (см. integrate_scan)
    int floor_confirmations = 3;     ///< столько раз клетку нужно увидеть свободной, чтобы считать
                                     ///< её «полом» (см. find_transient_hits)
    int hit_memory = 200;            ///< сколько попаданий помнит клетка (0…255). Наблюдение «свободно»
                                     ///< сначала гасит одно запомненное попадание и лишь потом снижает
                                     ///< оценку занятости: стену, которую робот много раз видел, не
                                     ///< стирают лучи, прошедшие вплотную вдоль неё (см. integrate_scan).
                                     ///< Запоминаются попадания в уже уверенно занятую клетку.
                                     ///< 0 — память выключена
};

enum class CellState : std::uint8_t { Free = 0, Occupied = 1, Unknown = 2 };

/// Отметки лучей, которые трекер подвижных объектов передаёт карте.
namespace ray_flag {
inline constexpr std::uint8_t Static = 0;   ///< обычное отражение: клетка помечается занятой
inline constexpr std::uint8_t Dynamic = 1;  ///< луч попал в подвижный объект: клетка отражения
                                            ///< занятой не помечается (путь до неё свободен)
inline constexpr std::uint8_t Settled = 2;  ///< объект, появившийся на «полу», признан неподвижным:
                                            ///< клетка помечается занятой и перестаёт быть «полом»
}  // namespace ray_flag

class OccupancyGrid {
 public:
    /// Карта размером config.size_x × config.size_y с центром в точке center.
    OccupancyGrid(const MapConfig& config, Vec2 center);

    /// Встраивает один скан.
    ///   sensor     — поза сенсора в СК карты;
    ///   dirs       — единичные направления лучей в СК сенсора;
    ///   ranges     — дальности [м]; значение >= range_max означает «отражения нет»;
    ///   ray_flags  — необязательные отметки лучей (см. ray_flag): Dynamic — клетка
    ///                отражения не помечается занятой (луч попал в подвижный
    ///                объект), но путь до неё по-прежнему считается свободным;
    ///                Settled — клетка помечается занятой, а её память «пола»
    ///                стирается (объект признан неподвижным и стал частью карты).
    /// За один скан каждая клетка обновляется не более одного раза, причём
    /// «занято» имеет приоритет над «свободно».
    ///
    /// Память попаданий. Прямое наблюдение препятствия (отражение) надёжнее
    /// косвенного (луч «прошёл сквозь клетку»): при сантиметровой ошибке позы луч,
    /// идущий вдоль грани стены, трассируется по клеткам самой стены. Поэтому
    /// каждая клетка помнит до hit_memory попаданий, и наблюдение «свободно»
    /// сначала гасит одно из них. Забывание остаётся, но становится тем медленнее,
    /// чем дольше клетка наблюдалась занятой. Запоминаются только попадания,
    /// подтверждающие уже уверенно занятую клетку (оценка на верхней границе):
    /// объект, проезжающий через клетку за несколько сканов, памяти не оставляет,
    /// и его след на карте исчезает без задержки.
    ///
    /// Угол падения. Луч, идущий почти вдоль стены, на последних метрах находится
    /// от неё в сантиметрах. Оценка позы всегда имеет погрешность того же порядка,
    /// поэтому такой луч легко «промахивается» и рисуется по клеткам самой стены,
    /// стирая её. Поэтому свободным помечается только участок луча, удалённый от
    /// поверхности больше чем на surface_clearance клеток: хвост длиной
    /// surface_clearance / sin(угла падения) пропускается. Угол падения
    /// оценивается по соседним точкам скана.
    void integrate_scan(const Pose2& sensor, std::span<const Vec2> dirs, std::span<const float> ranges,
                        double range_max, std::span<const std::uint8_t> ray_flags = {});

    /// Сравнение скана с картой: отмечает лучи, отражение которых пришло с «пола» —
    /// из места, которое раньше надёжно наблюдалось свободным (и сама клетка, и
    /// все её соседи). Стена так выглядеть не может: рядом с ней всегда есть
    /// клетки, свободными не бывавшие. Значит, это подвижный объект либо новый
    /// предмет. transient[i] = 1 для таких лучей.
    ///
    /// Используется именно память («бывала свободной»), а не текущее состояние
    /// клетки: объект, который медленно надвигается на сенсор, успевает пометить
    /// клетки перед собой занятыми, но полом они от этого быть не перестают.
    void find_transient_hits(const Pose2& sensor, std::span<const Vec2> dirs, std::span<const float> ranges,
                             double range_max, std::vector<std::uint8_t>& transient) const;

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    double resolution() const noexcept { return cfg_.resolution; }
    Vec2 origin() const noexcept { return origin_; }  ///< левый нижний угол клетки (0, 0)
    const MapConfig& config() const noexcept { return cfg_; }
    std::uint32_t scans() const noexcept { return scan_id_; }

    /// Прямоугольник клеток, которых касался хотя бы один луч (границы включительно).
    /// false — карта ещё пуста. Планировщику незачем обрабатывать остальное.
    bool observed_bounds(int& x0, int& y0, int& x1, int& y1) const noexcept {
        x0 = min_x_;
        y0 = min_y_;
        x1 = max_x_;
        y1 = max_y_;
        return max_x_ >= min_x_;
    }

    /// Клетка, содержащая точку. false — точка вне карты.
    bool cell_of(Vec2 p, int& ix, int& iy) const noexcept;
    Vec2 cell_center(int ix, int iy) const noexcept {
        return {origin_.x + (ix + 0.5) * cfg_.resolution, origin_.y + (iy + 0.5) * cfg_.resolution};
    }

    /// Лог-шансы клетки; вне карты — 0 (неизвестно).
    float log_odds(int ix, int iy) const noexcept {
        if (ix < 0 || iy < 0 || ix >= width_ || iy >= height_) return 0.0F;
        return log_odds_[static_cast<std::size_t>(iy) * width_ + ix];
    }
    CellState state(int ix, int iy) const noexcept {
        const float l = log_odds(ix, iy);
        if (l > cfg_.occupied_threshold) return CellState::Occupied;
        if (l < cfg_.free_threshold) return CellState::Free;
        return CellState::Unknown;
    }
    /// Вероятность занятости клетки p = 1 / (1 + exp(-l)).
    double probability(int ix, int iy) const noexcept;

    /// Лог-шансы всех клеток построчно (индекс = iy * width + ix).
    std::span<const float> data() const noexcept { return log_odds_; }

    /// Значение карты в произвольной точке, билинейная интерполяция по центрам
    /// клеток: +1 — уверенно занято, -1 — уверенно свободно, 0 — неизвестно.
    /// Гладкая функция положения — на ней работает сопоставление сканов.
    double occupancy_value(Vec2 p) const noexcept;

 private:
    void trace_free(double x0, double y0, double x1, double y1) noexcept;
    bool mark_free(int x, int y) noexcept;

    MapConfig cfg_;
    int width_;
    int height_;
    Vec2 origin_;
    std::vector<float> log_odds_;
    std::vector<std::uint8_t> floor_count_;  ///< сколько раз клетка подтверждалась свободной (с насыщением)
    std::vector<std::uint8_t> hit_count_;    ///< запомненные попадания (не больше hit_memory)
    // Отметка последнего обновления клетки: 2 * scan_id — «свободно», 2 * scan_id + 1 — «занято».
    std::vector<std::uint32_t> mark_;
    std::uint32_t scan_id_ = 0;
    int min_x_ = 0, min_y_ = 0, max_x_ = -1, max_y_ = -1;  ///< границы разведанной области
    // Концы свободных участков лучей текущего скана (в клетках) — для второго прохода.
    std::vector<double> end_x_, end_y_;
};

}  // namespace roversim::nav
