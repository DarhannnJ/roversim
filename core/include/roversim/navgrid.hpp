// Навигационная сетка: где может находиться центр робота и как далеко до цели.
//
// Нужна симулятору для трёх вещей:
//   * случайные старт и цель гарантированно достижимы друг из друга;
//   * награда за прогресс считается по длине кратчайшего пути, а не по прямой
//     (в лабиринте это принципиально: прямая упирается в стену);
//   * известна длина оптимального пути — база для метрики SPL.
#pragma once

#include <cstdint>
#include <vector>

#include "roversim/math.hpp"
#include "roversim/rng.hpp"
#include "roversim/world.hpp"

namespace roversim {

class NavGrid {
 public:
    /// Клетка проходима, если круг радиуса `radius` с центром в центре клетки
    /// свободен. Из всех связных областей оставляется наибольшая.
    NavGrid(const World& world, double radius);

    /// Число проходимых клеток (0 — роботу негде стоять).
    std::size_t free_count() const noexcept { return free_cells_.size(); }
    /// Центр случайной проходимой клетки.
    Vec2 sample(Rng& rng) const noexcept;

    /// Поле кратчайших расстояний [м] от каждой клетки до цели (Дейкстра,
    /// 8-связность без срезания углов). Недостижимые клетки = +inf.
    /// В `next` для каждой клетки записывается следующая клетка на кратчайшем
    /// пути к цели (-1 для самой цели и недостижимых клеток): по этим ссылкам
    /// путь восстанавливается без поиска.
    void distance_field(Vec2 goal, std::vector<float>& field, std::vector<std::int32_t>& next) const;

    /// Расстояние до цели из произвольной точки p: непрерывная интерполяция поля
    /// по соседним клеткам. +inf, если рядом нет проходимых клеток.
    double lookup(const std::vector<float>& field, Vec2 p) const noexcept;

    /// Точка в `ahead` метрах от start в направлении, куда ведёт кратчайший путь, —
    /// «морковка» для следования по маршруту. Направление берётся на самую дальнюю
    /// точку пути, видимую из start по прямой (в пределах нескольких `ahead`):
    /// сеточный путь идёт «лесенкой», а натянутая нить вдоль него — прямыми
    /// отрезками от угла к углу. Без выделения памяти.
    Vec2 lookahead(const std::vector<std::int32_t>& next, Vec2 start, Vec2 goal, double ahead) const noexcept;
    /// Отрезок целиком лежит в проходимых клетках.
    bool line_free(Vec2 a, Vec2 b) const noexcept;

    /// Кратчайший путь от start до цели, для которой построено поле.
    /// Пусто, если цель недостижима.
    std::vector<Vec2> path(const std::vector<std::int32_t>& next, Vec2 start, Vec2 goal) const;

 private:
    int nearest_free(Vec2 p) const noexcept;  ///< ближайшая проходимая клетка рядом с p или -1
    Vec2 center(int index) const noexcept {
        return {origin_.x + (index % width_ + 0.5) * cell_, origin_.y + (index / width_ + 0.5) * cell_};
    }

    int width_;
    int height_;
    double cell_;
    Vec2 origin_;
    std::vector<std::uint8_t> free_;      ///< 1 — клетка проходима
    std::vector<std::int32_t> free_cells_;  ///< индексы проходимых клеток (для выборки)
};

}  // namespace roversim
