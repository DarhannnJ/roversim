// Глобальный планировщик маршрута на сетке: A* и Theta*.
//
// Карта препятствий превращается в карту стоимости (costmap):
//   * клетки ближе к препятствию, чем радиус робота с запасом, запрещены;
//   * в «мягкой» зоне вокруг них движение дорожает — маршрут держится середины
//     проходов, а не прижимается к стенам;
//   * неразведанные клетки по умолчанию проходимы (оптимистичное планирование:
//     робот едет к цели и перепланирует, когда лидар открывает новые стены).
//
// A* ищет путь по 8-связной сетке; Theta* — его вариант «под любым углом»:
// родителем клетки может быть любая клетка в прямой видимости, поэтому путь
// состоит из длинных прямых отрезков и короче сеточного.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "roversim/math.hpp"
#include "roversim/nav/occupancy_grid.hpp"
#include "roversim/world.hpp"

namespace roversim::nav {

/// Точное евклидово преобразование расстояний (Felzenszwalb, Huttenlocher).
/// occupied — маска [height * width]; distance[i] — расстояние от центра клетки i
/// до центра ближайшей занятой клетки, в клетках (очень большое, если занятых нет).
/// Сложность линейна по числу клеток.
void distance_transform(std::span<const std::uint8_t> occupied, int width, int height, std::vector<float>& distance);

struct PlannerConfig {
    enum class Algorithm { AStar, ThetaStar };

    Algorithm algorithm = Algorithm::ThetaStar;
    double robot_radius = 0.20;    ///< радиус робота [м]
    double safety_margin = 0.06;   ///< запас сверх радиуса: ближе к препятствиям маршрут не подходит [м]
    double soft_radius = 0.50;     ///< граница зоны удорожания, расстояние от препятствия [м]
    double soft_weight = 4.0;      ///< во сколько раз (сверх единицы) дороже путь у края запретной зоны
    double unknown_penalty = 0.5;  ///< добавка к стоимости неразведанных клеток
    bool allow_unknown = true;     ///< можно ли прокладывать путь через неразведанное
    bool smooth = true;            ///< спрямлять путь A* по прямой видимости
    int max_expansions = 400000;   ///< предел числа раскрытых клеток
};

class GridPlanner {
 public:
    explicit GridPlanner(const PlannerConfig& config);

    /// Строит карту стоимости по карте занятости SLAM.
    void set_map(const OccupancyGrid& map);
    /// То же, но только по окну: разведанная область плюс точки a и b (обычно
    /// старт и цель) с запасом по краям. На большой, едва разведанной карте это
    /// в разы дешевле: стоимость растёт с площадью окна, а не всей карты.
    void set_map(const OccupancyGrid& map, Vec2 a, Vec2 b);
    /// То же по истинной карте мира (для экспериментов с известной картой).
    void set_map(const World& world);
    void set_cells(std::span<const CellState> cells, int width, int height, double resolution, Vec2 origin);

    /// Ищет путь от start до goal. Возвращает false, если пути нет.
    /// Если старт или цель оказались в запретной зоне (робот у самой стены),
    /// поиск ведётся от ближайшей разрешённой клетки.
    bool plan(Vec2 start, Vec2 goal, std::vector<Vec2>& path);

    /// Расстояние от точки до ближайшего препятствия [м] (до границы занятой клетки).
    double clearance(Vec2 p) const noexcept;
    /// Отрезок не пересекает запретных клеток.
    bool line_free(Vec2 a, Vec2 b) const noexcept;

    bool has_map() const noexcept { return width_ > 0; }
    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    /// Множитель стоимости каждой клетки; 0 — клетка запрещена.
    std::span<const float> cost() const noexcept { return cost_; }
    /// Расстояние до ближайшего препятствия для каждой клетки [м].
    std::span<const float> clearance_field() const noexcept { return clearance_; }
    /// Сколько клеток раскрыл последний поиск.
    std::size_t expansions() const noexcept { return expansions_; }
    /// Стоимость последнего найденного пути (длина с учётом удорожания у стен).
    double path_cost() const noexcept { return path_cost_; }
    const PlannerConfig& config() const noexcept { return cfg_; }

 private:
    int cell_index(Vec2 p) const noexcept;
    Vec2 cell_center(int index) const noexcept {
        return {origin_.x + (index % width_ + 0.5) * resolution_, origin_.y + (index / width_ + 0.5) * resolution_};
    }
    int nearest_traversable(int cell) const noexcept;
    bool line_of_sight(int from, int to, double* cost) const noexcept;
    void rebuild_cost(std::span<const CellState> cells);

    PlannerConfig cfg_;
    int width_ = 0;
    int height_ = 0;
    double resolution_ = 0.0;
    Vec2 origin_{};
    std::vector<std::uint8_t> occupied_;
    std::vector<CellState> cells_;
    std::vector<float> clearance_;
    std::vector<float> cost_;

    // Состояние поиска. Массивы не очищаются между вызовами: актуальность записи
    // определяется по номеру поколения.
    std::vector<float> g_;
    std::vector<std::int32_t> parent_;
    std::vector<std::uint32_t> seen_, closed_;
    std::uint32_t generation_ = 0;
    std::size_t expansions_ = 0;
    double path_cost_ = 0.0;
};

}  // namespace roversim::nav
