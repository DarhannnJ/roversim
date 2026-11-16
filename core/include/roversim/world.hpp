// Статический мир: сетка занятости + аналитические препятствия (AABB, круги).
//
// Один и тот же объект World отвечает на два вида запросов:
//   * raycast  — для лидара (DDA по сетке с ускорением по полю расстояний);
//   * collide  — для физики (круг робота против клеток, AABB и кругов).
// После finalize() все запросы const и потокобезопасны, поэтому один World
// можно разделять между любым числом симуляторов (см. VecSimulator).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "roversim/math.hpp"

namespace roversim {

struct Aabb {
    Vec2 min;
    Vec2 max;
};

struct Circle {
    Vec2 center;
    double radius = 0.0;
};

/// Контакт «круг робота — препятствие».
struct Contact {
    Vec2 normal;       ///< единичная нормаль от препятствия к роботу (мировая СК)
    double gap = 0.0;  ///< зазор [м]; отрицательный — глубина проникновения
};

/// Расстояние вдоль луча (origin, единичный dir) до круга; t_max при промахе.
inline double ray_circle_distance(Vec2 origin, Vec2 dir, const Circle& circle, double t_max) noexcept {
    const Vec2 m = circle.center - origin;
    const double cc = dot(m, m) - circle.radius * circle.radius;
    if (cc <= 0.0) return 0.0;  // начало луча внутри круга
    const double b = dot(m, dir);
    if (b <= 0.0) return t_max;  // круг позади
    const double disc = b * b - cc;
    if (disc < 0.0) return t_max;
    const double t = b - std::sqrt(disc);
    return t < t_max ? t : t_max;
}

class World {
 public:
    /// Пустая сетка width × height клеток со стороной cell_size [м];
    /// origin — мировые координаты левого нижнего угла клетки (0, 0).
    World(int width, int height, double cell_size, Vec2 origin = {});

    // ---- Редактирование. После любых правок нужен finalize(). ----
    void set_cell(int ix, int iy, bool occupied);
    /// Растеризует прямоугольник в сетку: помечает все клетки, которые он задевает.
    void fill_box(const Aabb& box, bool occupied = true);
    /// Аналитические препятствия: точная геометрия, не привязанная к сетке.
    void add_box(const Aabb& box);
    void add_circle(const Circle& circle);

    /// Строит ускоряющую структуру (поле расстояний Чебышёва). Идемпотентно.
    void finalize();
    bool finalized() const noexcept { return finalized_; }
    /// Номер ревизии: растёт при каждом finalize(). По нему симулятор замечает,
    /// что мир изменился, и перестраивает зависящие от него данные.
    std::uint64_t revision() const noexcept { return revision_; }

    // ---- Запросы (только после finalize()). ----

    /// Расстояние вдоль луча до первого препятствия, но не более max_range.
    /// dir — единичный вектор. Край сетки считается стеной.
    double raycast(Vec2 origin, Vec2 dir, double max_range) const noexcept;

    /// Пакет лучей из одной точки (скан лидара). Направления заданы в СК сенсора
    /// и поворачиваются на курс (cos_yaw, sin_yaw). Аналитические примитивы
    /// отбираются по дальности один раз на весь скан.
    void raycast_scan(Vec2 origin, double cos_yaw, double sin_yaw,
                      std::span<const Vec2> local_dirs, double max_range,
                      std::span<float> out) const noexcept;

    /// Контакты круга (center, radius) с препятствиями, зазор до которых меньше
    /// margin. Возвращает число записанных в out контактов.
    std::size_t collide_circle(Vec2 center, double radius, double margin,
                               std::span<Contact> out) const noexcept;

    /// true, если круг не пересекает ни одно препятствие и лежит внутри мира.
    bool is_free(Vec2 center, double radius) const noexcept;

    // ---- Доступ к данным. ----
    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    double cell_size() const noexcept { return cell_; }
    Vec2 origin() const noexcept { return origin_; }
    Aabb bounds() const noexcept {
        return {origin_, {origin_.x + width_ * cell_, origin_.y + height_ * cell_}};
    }
    /// Клетка занята? За пределами сетки — всегда true (мир окружён стеной).
    bool occupied(int ix, int iy) const noexcept {
        if (static_cast<unsigned>(ix) >= static_cast<unsigned>(width_) ||
            static_cast<unsigned>(iy) >= static_cast<unsigned>(height_)) {
            return true;
        }
        return occ_[static_cast<std::size_t>(iy) * width_ + ix] != 0;
    }
    Vec2 cell_center(int ix, int iy) const noexcept {
        return {origin_.x + (ix + 0.5) * cell_, origin_.y + (iy + 0.5) * cell_};
    }
    /// Сетка занятости, построчно (индекс = iy * width + ix), 0/1.
    std::span<const std::uint8_t> cells() const noexcept { return occ_; }
    std::span<const Aabb> boxes() const noexcept { return boxes_; }
    std::span<const Circle> circles() const noexcept { return circles_; }

    // ---- Генераторы готовых (уже финализированных) миров. ----

    /// Пустая арена width_m × height_m со стенами по периметру.
    static World arena(double width_m, double height_m, double cell_size = 0.05);

    /// Лабиринт cols × rows ячеек (обход в глубину). corridor_m — ширина прохода,
    /// wall_m — толщина стен. loop_prob > 0 дополнительно пробивает часть стен,
    /// создавая петли (несколько путей к цели).
    static World maze(int cols, int rows, double corridor_m, double wall_m, double cell_size,
                      std::uint64_t seed, double loop_prob = 0.0);

    /// Арена со случайными препятствиями: num_boxes прямоугольников (в сетке)
    /// и num_pillars круглых колонн (аналитических).
    static World cluttered(double width_m, double height_m, double cell_size, int num_boxes,
                           int num_pillars, std::uint64_t seed);

    /// Карта из текста: '#' — стена, всё остальное — свободно. Первая строка — верх карты.
    static World from_ascii(std::span<const std::string_view> rows, double cell_size);

 private:
    double raycast_grid(Vec2 origin, Vec2 dir, double max_range) const noexcept;
    void touch() noexcept { finalized_ = false; }

    int width_;
    int height_;
    double cell_;
    double inv_cell_;
    Vec2 origin_;
    std::vector<std::uint8_t> occ_;    ///< занятость 0/1
    std::vector<std::uint8_t> clear_;  ///< расстояние Чебышёва до ближайшей занятой клетки
                                       ///< (в клетках, 0 = занята, насыщение на 255)
    std::vector<Aabb> boxes_;
    std::vector<Circle> circles_;
    bool finalized_ = false;
    std::uint64_t revision_ = 0;
};

}  // namespace roversim
