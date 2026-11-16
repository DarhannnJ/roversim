#include "roversim/world.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "roversim/rng.hpp"

namespace roversim {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Минимальный clearance (в клетках), при котором вместо шага DDA делается «прыжок»
// по полю расстояний. Значение подобрано замером (см. roversim_bench).
constexpr int kJumpMin = 2;
// Запас прыжка в клетках: не даёт ошибке округления вынести точку приземления
// за пределы гарантированно свободной области.
constexpr double kJumpSlack = 1e-6;

/// Луч против AABB (метод слоёв). Возвращает расстояние до входа или t_max.
inline double ray_box(Vec2 o, Vec2 d, const Aabb& b, double t_max) noexcept {
    double t0 = 0.0, t1 = t_max;
    const auto slab = [&](double origin, double dir, double lo, double hi) {
        if (dir != 0.0) {
            const double inv = 1.0 / dir;
            double ta = (lo - origin) * inv, tb = (hi - origin) * inv;
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
            return true;
        }
        return origin >= lo && origin <= hi;  // луч параллелен слою
    };
    if (!slab(o.x, d.x, b.min.x, b.max.x) || !slab(o.y, d.y, b.min.y, b.max.y)) return t_max;
    return t0 <= t1 ? t0 : t_max;
}

inline double point_box_distance_sq(Vec2 p, const Aabb& b) noexcept {
    const double dx = std::max({b.min.x - p.x, 0.0, p.x - b.max.x});
    const double dy = std::max({b.min.y - p.y, 0.0, p.y - b.max.y});
    return dx * dx + dy * dy;
}

/// Сетка со стенами по периметру (ещё не финализированная).
World bordered_grid(double width_m, double height_m, double cell_size) {
    if (!(cell_size > 0.0)) throw std::invalid_argument("World: cell_size must be positive");
    const int w = std::max(3, static_cast<int>(std::lround(width_m / cell_size)));
    const int h = std::max(3, static_cast<int>(std::lround(height_m / cell_size)));
    World world(w, h, cell_size);
    for (int ix = 0; ix < w; ++ix) {
        world.set_cell(ix, 0, true);
        world.set_cell(ix, h - 1, true);
    }
    for (int iy = 0; iy < h; ++iy) {
        world.set_cell(0, iy, true);
        world.set_cell(w - 1, iy, true);
    }
    return world;
}

}  // namespace

World::World(int width, int height, double cell_size, Vec2 origin)
    : width_(width), height_(height), cell_(cell_size), inv_cell_(1.0 / cell_size), origin_(origin) {
    if (width <= 0 || height <= 0 || !(cell_size > 0.0)) {
        throw std::invalid_argument("World: width, height and cell_size must be positive");
    }
    if (static_cast<long long>(width) * height > (1LL << 28)) {
        throw std::invalid_argument("World: grid is too large");
    }
    const auto size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    occ_.assign(size, 0);
    clear_.assign(size, 0);
}

void World::set_cell(int ix, int iy, bool occupied) {
    if (ix < 0 || iy < 0 || ix >= width_ || iy >= height_) {
        throw std::out_of_range("World::set_cell: cell index out of range");
    }
    occ_[static_cast<std::size_t>(iy) * width_ + ix] = occupied ? 1 : 0;
    touch();
}

void World::fill_box(const Aabb& box, bool occupied) {
    // Допуск 1e-9 клетки: прямоугольник, лежащий ровно по границам клеток,
    // не должен из-за округления захватывать соседний ряд.
    const int ix0 = std::max(0, static_cast<int>(std::floor((box.min.x - origin_.x) * inv_cell_ + 1e-9)));
    const int iy0 = std::max(0, static_cast<int>(std::floor((box.min.y - origin_.y) * inv_cell_ + 1e-9)));
    const int ix1 = std::min(width_, static_cast<int>(std::ceil((box.max.x - origin_.x) * inv_cell_ - 1e-9)));
    const int iy1 = std::min(height_, static_cast<int>(std::ceil((box.max.y - origin_.y) * inv_cell_ - 1e-9)));
    for (int iy = iy0; iy < iy1; ++iy) {
        for (int ix = ix0; ix < ix1; ++ix) {
            occ_[static_cast<std::size_t>(iy) * width_ + ix] = occupied ? 1 : 0;
        }
    }
    touch();
}

void World::add_box(const Aabb& box) {
    if (!(box.max.x > box.min.x) || !(box.max.y > box.min.y)) {
        throw std::invalid_argument("World::add_box: box must have positive size");
    }
    boxes_.push_back(box);
    touch();
}

void World::add_circle(const Circle& circle) {
    if (!(circle.radius > 0.0)) throw std::invalid_argument("World::add_circle: radius must be positive");
    circles_.push_back(circle);
    touch();
}

void World::finalize() {
    if (finalized_) return;
    // Двухпроходное преобразование расстояний в метрике Чебышёва (для неё оно
    // точное). Всё, что за краем сетки, считается занятым, поэтому «прыжок» в
    // raycast_grid никогда не выводит луч за пределы массива.
    const int w = width_, h = height_;
    const auto at = [&](int ix, int iy) -> int {
        if (ix < 0 || iy < 0 || ix >= w || iy >= h) return 0;
        return clear_[static_cast<std::size_t>(iy) * w + ix];
    };
    for (int iy = 0; iy < h; ++iy) {
        for (int ix = 0; ix < w; ++ix) {
            const std::size_t i = static_cast<std::size_t>(iy) * w + ix;
            if (occ_[i] != 0) {
                clear_[i] = 0;
                continue;
            }
            const int m = std::min({at(ix - 1, iy), at(ix - 1, iy - 1), at(ix, iy - 1), at(ix + 1, iy - 1)});
            clear_[i] = static_cast<std::uint8_t>(std::min(m + 1, 255));
        }
    }
    for (int iy = h - 1; iy >= 0; --iy) {
        for (int ix = w - 1; ix >= 0; --ix) {
            const std::size_t i = static_cast<std::size_t>(iy) * w + ix;
            const int m = std::min({at(ix + 1, iy), at(ix + 1, iy + 1), at(ix, iy + 1), at(ix - 1, iy + 1)});
            clear_[i] = static_cast<std::uint8_t>(std::min<int>(clear_[i], std::min(m + 1, 255)));
        }
    }
    finalized_ = true;
    ++revision_;
}

// Трассировка луча по сетке: алгоритм Аманатидеса — Ву (DDA) с ускорением.
//
// Обычный DDA посещает каждую клетку на пути луча. Здесь у каждой свободной
// клетки записано расстояние Чебышёва k до ближайшей занятой, а значит, из любой
// её точки можно без проверок пройти (k - 1) клеток евклидова расстояния: каждая
// координата изменится меньше чем на k клеток. На открытых участках луч
// идёт большими прыжками, а вблизи стен алгоритм переходит на точные шаги DDA,
// так что дистанция до попадания вычисляется по границе клетки без погрешности.
double World::raycast_grid(Vec2 o, Vec2 d, double max_range) const noexcept {
    const int w = width_, h = height_;
    // Дальше все длины измеряются в клетках.
    const double px = (o.x - origin_.x) * inv_cell_;
    const double py = (o.y - origin_.y) * inv_cell_;
    if (!(px >= 0.0 && py >= 0.0 && px < w && py < h)) return 0.0;  // снаружи = внутри стены

    int ix = static_cast<int>(px);
    int iy = static_cast<int>(py);
    const std::uint8_t* clr = clear_.data();
    if (clr[static_cast<std::size_t>(iy) * w + ix] == 0) return 0.0;

    const double t_end = max_range * inv_cell_;
    const int step_x = d.x > 0.0 ? 1 : -1;
    const int step_y = d.y > 0.0 ? 1 : -1;
    const double inv_dx = d.x != 0.0 ? 1.0 / d.x : kInf;
    const double inv_dy = d.y != 0.0 ? 1.0 / d.y : kInf;
    const double t_delta_x = std::abs(inv_dx);
    const double t_delta_y = std::abs(inv_dy);
    // Параметр луча, при котором он пересекает ближайшую границу клетки по каждой оси.
    const auto next_x = [&](int cell) { return d.x != 0.0 ? ((d.x > 0.0 ? cell + 1 : cell) - px) * inv_dx : kInf; };
    const auto next_y = [&](int cell) { return d.y != 0.0 ? ((d.y > 0.0 ? cell + 1 : cell) - py) * inv_dy : kInf; };
    double t_max_x = next_x(ix);
    double t_max_y = next_y(iy);
    double t = 0.0;

    for (;;) {
        const int clearance = clr[static_cast<std::size_t>(iy) * w + ix];
        if (clearance >= kJumpMin) {
            t += (clearance - 1) - kJumpSlack;
            if (t >= t_end) return max_range;
            ix = static_cast<int>(px + t * d.x);
            iy = static_cast<int>(py + t * d.y);
            t_max_x = next_x(ix);
            t_max_y = next_y(iy);
            continue;
        }
        if (t_max_x < t_max_y) {
            t = t_max_x;
            t_max_x += t_delta_x;
            ix += step_x;
        } else {
            t = t_max_y;
            t_max_y += t_delta_y;
            iy += step_y;
        }
        if (t >= t_end) return max_range;
        if (static_cast<unsigned>(ix) >= static_cast<unsigned>(w) ||
            static_cast<unsigned>(iy) >= static_cast<unsigned>(h)) {
            return t * cell_;  // край мира — стена
        }
        if (clr[static_cast<std::size_t>(iy) * w + ix] == 0) return t * cell_;
    }
}

double World::raycast(Vec2 origin, Vec2 dir, double max_range) const noexcept {
    double t = raycast_grid(origin, dir, max_range);
    for (const Aabb& b : boxes_) t = ray_box(origin, dir, b, t);
    for (const Circle& c : circles_) t = ray_circle_distance(origin, dir, c, t);
    return t;
}

void World::raycast_scan(Vec2 origin, double cos_yaw, double sin_yaw, std::span<const Vec2> local_dirs,
                         double max_range, std::span<float> out) const noexcept {
    // Аналитические примитивы проверяются перебором, поэтому сначала оставляем
    // только те, до которых сенсор в принципе может дотянуться. Буферы
    // thread_local: после прогрева вызов не выделяет память и остаётся потокобезопасным.
    thread_local std::vector<Aabb> near_boxes;
    thread_local std::vector<Circle> near_circles;
    near_boxes.clear();
    near_circles.clear();
    for (const Aabb& b : boxes_) {
        if (point_box_distance_sq(origin, b) < max_range * max_range) near_boxes.push_back(b);
    }
    for (const Circle& c : circles_) {
        if (distance(origin, c.center) < max_range + c.radius) near_circles.push_back(c);
    }

    const std::size_t n = std::min(local_dirs.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) {
        const Vec2 dir = rotate(local_dirs[i], cos_yaw, sin_yaw);
        double t = raycast_grid(origin, dir, max_range);
        for (const Aabb& b : near_boxes) t = ray_box(origin, dir, b, t);
        for (const Circle& c : near_circles) t = ray_circle_distance(origin, dir, c, t);
        out[i] = static_cast<float>(t);
    }
}

std::size_t World::collide_circle(Vec2 c, double r, double margin, std::span<Contact> out) const noexcept {
    std::size_t n = 0;
    const auto emit = [&](Vec2 normal, double gap) {
        if (gap < margin && n < out.size()) out[n++] = {normal, gap};
    };

    // 1. Границы мира: четыре полуплоскости.
    const Aabb world_box = bounds();
    emit({1.0, 0.0}, c.x - r - world_box.min.x);
    emit({-1.0, 0.0}, world_box.max.x - c.x - r);
    emit({0.0, 1.0}, c.y - r - world_box.min.y);
    emit({0.0, -1.0}, world_box.max.y - c.y - r);

    // 2. Клетки сетки. Поле расстояний даёт дешёвый ранний выход: если ближайшая
    //    занятая клетка заведомо дальше r + margin, перебор соседей не нужен.
    const double reach = r + margin;
    const double gx = (c.x - origin_.x) * inv_cell_;
    const double gy = (c.y - origin_.y) * inv_cell_;
    const int cx = std::clamp(static_cast<int>(std::floor(gx)), 0, width_ - 1);
    const int cy = std::clamp(static_cast<int>(std::floor(gy)), 0, height_ - 1);
    const int clearance = clear_[static_cast<std::size_t>(cy) * width_ + cx];
    if ((clearance - 1) * cell_ <= reach) {
        const double span = reach * inv_cell_;
        const int ix0 = std::clamp(static_cast<int>(std::floor(gx - span)), 0, width_ - 1);
        const int ix1 = std::clamp(static_cast<int>(std::floor(gx + span)), 0, width_ - 1);
        const int iy0 = std::clamp(static_cast<int>(std::floor(gy - span)), 0, height_ - 1);
        const int iy1 = std::clamp(static_cast<int>(std::floor(gy + span)), 0, height_ - 1);
        for (int iy = iy0; iy <= iy1; ++iy) {
            for (int ix = ix0; ix <= ix1; ++ix) {
                if (occ_[static_cast<std::size_t>(iy) * width_ + ix] == 0) continue;
                const double x0 = origin_.x + ix * cell_, x1 = x0 + cell_;
                const double y0 = origin_.y + iy * cell_, y1 = y0 + cell_;
                // В какой зоне Вороного клетки лежит центр круга: -1 / 0 / +1 по каждой оси.
                const int sx = c.x < x0 ? -1 : (c.x >= x1 ? 1 : 0);
                const int sy = c.y < y0 ? -1 : (c.y >= y1 ? 1 : 0);

                // Стена из нескольких клеток — одно тело. Контакты с гранями и
                // вершинами, скрытыми между соседними занятыми клетками, отбрасываются:
                // иначе круг, скользящий вдоль ровной стены, «цеплялся» бы за стыки.
                if (sx != 0 && sy != 0) {
                    if (occupied(ix + sx, iy) || occupied(ix, iy + sy)) continue;
                    const Vec2 corner{sx < 0 ? x0 : x1, sy < 0 ? y0 : y1};
                    const Vec2 delta = c - corner;
                    const double dist = norm(delta);
                    const Vec2 normal = dist > 1e-12
                                            ? delta * (1.0 / dist)
                                            : Vec2{sx * std::numbers::sqrt2 / 2.0, sy * std::numbers::sqrt2 / 2.0};
                    emit(normal, dist - r);
                } else if (sx != 0) {
                    if (occupied(ix + sx, iy)) continue;
                    emit({static_cast<double>(sx), 0.0}, (sx < 0 ? x0 - c.x : c.x - x1) - r);
                } else if (sy != 0) {
                    if (occupied(ix, iy + sy)) continue;
                    emit({0.0, static_cast<double>(sy)}, (sy < 0 ? y0 - c.y : c.y - y1) - r);
                } else {
                    // Центр внутри занятой клетки (в штатной работе не бывает):
                    // выталкиваем через ближайшую открытую грань.
                    double best = kInf;
                    Vec2 normal{1.0, 0.0};
                    const auto consider = [&](double depth, Vec2 face_normal, bool open) {
                        if (open && depth < best) {
                            best = depth;
                            normal = face_normal;
                        }
                    };
                    consider(c.x - x0, {-1.0, 0.0}, !occupied(ix - 1, iy));
                    consider(x1 - c.x, {1.0, 0.0}, !occupied(ix + 1, iy));
                    consider(c.y - y0, {0.0, -1.0}, !occupied(ix, iy - 1));
                    consider(y1 - c.y, {0.0, 1.0}, !occupied(ix, iy + 1));
                    if (best == kInf) best = cell_;  // замурован со всех сторон
                    emit(normal, -(best + r));
                }
            }
        }
    }

    // 3. Аналитические прямоугольники.
    for (const Aabb& b : boxes_) {
        const Vec2 closest{std::clamp(c.x, b.min.x, b.max.x), std::clamp(c.y, b.min.y, b.max.y)};
        const Vec2 delta = c - closest;
        const double d2 = dot(delta, delta);
        if (d2 > 0.0) {
            const double dist = std::sqrt(d2);
            emit(delta * (1.0 / dist), dist - r);
        } else {
            // Центр внутри: выталкиваем вдоль оси с наименьшей глубиной.
            double best = kInf;
            Vec2 normal{1.0, 0.0};
            const auto consider = [&](double depth, Vec2 face_normal) {
                if (depth < best) {
                    best = depth;
                    normal = face_normal;
                }
            };
            consider(c.x - b.min.x, {-1.0, 0.0});
            consider(b.max.x - c.x, {1.0, 0.0});
            consider(c.y - b.min.y, {0.0, -1.0});
            consider(b.max.y - c.y, {0.0, 1.0});
            emit(normal, -(best + r));
        }
    }

    // 4. Аналитические круги.
    for (const Circle& obstacle : circles_) {
        const Vec2 delta = c - obstacle.center;
        const double dist = norm(delta);
        const Vec2 normal = dist > 1e-12 ? delta * (1.0 / dist) : Vec2{1.0, 0.0};
        emit(normal, dist - r - obstacle.radius);
    }
    return n;
}

bool World::is_free(Vec2 center, double radius) const noexcept {
    Contact scratch[4];
    return collide_circle(center, radius, 0.0, scratch) == 0;
}

// ------------------------------------------------------------------ генераторы

World World::arena(double width_m, double height_m, double cell_size) {
    World world = bordered_grid(width_m, height_m, cell_size);
    world.finalize();
    return world;
}

World World::maze(int cols, int rows, double corridor_m, double wall_m, double cell_size,
                  std::uint64_t seed, double loop_prob) {
    if (cols < 1 || rows < 1) throw std::invalid_argument("World::maze: cols and rows must be >= 1");
    if (!(cell_size > 0.0)) throw std::invalid_argument("World::maze: cell_size must be positive");
    const int cw = std::max(1, static_cast<int>(std::lround(corridor_m / cell_size)));  // проход, клеток
    const int ww = std::max(1, static_cast<int>(std::lround(wall_m / cell_size)));      // стена, клеток
    const int pitch = cw + ww;
    World world(cols * pitch + ww, rows * pitch + ww, cell_size);
    for (int iy = 0; iy < world.height(); ++iy) {
        for (int ix = 0; ix < world.width(); ++ix) world.set_cell(ix, iy, true);
    }

    const auto carve = [&](int x0, int y0, int nx, int ny) {
        for (int iy = y0; iy < y0 + ny; ++iy) {
            for (int ix = x0; ix < x0 + nx; ++ix) world.set_cell(ix, iy, false);
        }
    };
    const auto cell_x0 = [&](int c) { return ww + c * pitch; };  // левый/нижний край ячейки
    const auto carve_cell = [&](int c, int r) { carve(cell_x0(c), cell_x0(r), cw, cw); };
    // Пробивает стену между ячейкой (c, r) и её соседом справа / сверху.
    const auto open_right = [&](int c, int r) { carve(cell_x0(c) + cw, cell_x0(r), ww, cw); };
    const auto open_up = [&](int c, int r) { carve(cell_x0(c), cell_x0(r) + cw, cw, ww); };
    const auto open_between = [&](int c0, int r0, int c1, int r1) {
        if (c0 != c1) {
            open_right(std::min(c0, c1), r0);
        } else {
            open_up(c0, std::min(r0, r1));
        }
    };

    // «Идеальный» лабиринт: итеративный обход в глубину со случайным выбором соседа.
    Rng rng = Rng::stream(seed, 0x4D415A45);  // 'MAZE'
    std::vector<std::uint8_t> visited(static_cast<std::size_t>(cols) * rows, 0);
    std::vector<int> stack;
    stack.push_back(0);
    visited[0] = 1;
    carve_cell(0, 0);
    constexpr int kDx[4] = {1, -1, 0, 0};
    constexpr int kDy[4] = {0, 0, 1, -1};
    while (!stack.empty()) {
        const int cur = stack.back();
        const int c = cur % cols, r = cur / cols;
        int options[4];
        int count = 0;
        for (int k = 0; k < 4; ++k) {
            const int nc = c + kDx[k], nr = r + kDy[k];
            if (nc < 0 || nr < 0 || nc >= cols || nr >= rows) continue;
            if (visited[static_cast<std::size_t>(nr) * cols + nc] == 0) options[count++] = k;
        }
        if (count == 0) {
            stack.pop_back();
            continue;
        }
        const int k = options[rng.below(static_cast<std::uint64_t>(count))];
        const int nc = c + kDx[k], nr = r + kDy[k];
        open_between(c, r, nc, nr);
        carve_cell(nc, nr);
        visited[static_cast<std::size_t>(nr) * cols + nc] = 1;
        stack.push_back(nr * cols + nc);
    }

    // Петли: пробиваем часть уцелевших внутренних стен.
    if (loop_prob > 0.0) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (c + 1 < cols && world.occupied(cell_x0(c) + cw, cell_x0(r)) && rng.uniform() < loop_prob) {
                    open_right(c, r);
                }
                if (r + 1 < rows && world.occupied(cell_x0(c), cell_x0(r) + cw) && rng.uniform() < loop_prob) {
                    open_up(c, r);
                }
            }
        }
    }
    world.finalize();
    return world;
}

World World::cluttered(double width_m, double height_m, double cell_size, int num_boxes, int num_pillars,
                       std::uint64_t seed) {
    World world = bordered_grid(width_m, height_m, cell_size);
    Rng rng = Rng::stream(seed, 0x434C5554);  // 'CLUT'
    const Aabb b = world.bounds();
    for (int i = 0; i < num_boxes; ++i) {
        const double sx = rng.uniform(0.3, 1.2), sy = rng.uniform(0.3, 1.2);
        const double x = rng.uniform(b.min.x, b.max.x - sx), y = rng.uniform(b.min.y, b.max.y - sy);
        world.fill_box({{x, y}, {x + sx, y + sy}});
    }
    for (int i = 0; i < num_pillars; ++i) {
        const double radius = rng.uniform(0.10, 0.35);
        world.add_circle({{rng.uniform(b.min.x, b.max.x), rng.uniform(b.min.y, b.max.y)}, radius});
    }
    world.finalize();
    return world;
}

World World::from_ascii(std::span<const std::string_view> rows, double cell_size) {
    std::size_t width = 0;
    for (std::string_view row : rows) width = std::max(width, row.size());
    if (rows.empty() || width == 0) throw std::invalid_argument("World::from_ascii: empty map");
    const int h = static_cast<int>(rows.size());
    World world(static_cast<int>(width), h, cell_size);
    for (int r = 0; r < h; ++r) {
        const std::string_view row = rows[static_cast<std::size_t>(r)];
        for (std::size_t ix = 0; ix < row.size(); ++ix) {
            if (row[ix] == '#') world.set_cell(static_cast<int>(ix), h - 1 - r, true);
        }
    }
    world.finalize();
    return world;
}

}  // namespace roversim
