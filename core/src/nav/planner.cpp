#include "roversim/nav/planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace roversim::nav {
namespace {

// «Бесконечность» для квадратов расстояний. Конечная, чтобы в формулах не
// возникало inf - inf.
constexpr double kFar = 1e12;

constexpr int kDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
constexpr int kDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

// Одномерное преобразование расстояний: d[q] = min_p ((q - p)^2 + f[p]).
// Каждое p задаёт параболу; ответ — нижняя огибающая всех парабол. Она строится
// за один проход: v хранит вершины парабол огибающей, z — границы их участков.
void transform_1d(const double* f, int n, double* d, int* v, double* z) {
    // Абсцисса пересечения парабол с вершинами p и q. По модулю она меньше kFar,
    // поэтому сравнение с z[0] = -kFar всегда останавливает цикл ниже.
    const auto intersection = [&](int p, int q) {
        return ((f[q] + static_cast<double>(q) * q) - (f[p] + static_cast<double>(p) * p)) / (2.0 * (q - p));
    };
    int k = 0;
    v[0] = 0;
    z[0] = -kFar;
    z[1] = kFar;
    for (int q = 1; q < n; ++q) {
        double s = intersection(v[k], q);
        while (s <= z[k]) {  // новая парабола закрывает последнюю из огибающей
            --k;
            s = intersection(v[k], q);
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kFar;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        const int p = v[k];
        d[q] = static_cast<double>(q - p) * (q - p) + f[p];
    }
}

}  // namespace

void distance_transform(std::span<const std::uint8_t> occupied, int width, int height, std::vector<float>& distance) {
    const auto size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (occupied.size() != size) throw std::invalid_argument("distance_transform: mask size mismatch");
    distance.resize(size);

    // Буферы thread_local: преобразование вызывается на каждом шаге локального
    // планировщика и не должно выделять память.
    thread_local std::vector<double> f, d, z, columns;
    thread_local std::vector<int> v;
    const auto longest = static_cast<std::size_t>(std::max(width, height));
    f.resize(longest);
    d.resize(longest);
    z.resize(longest + 1);
    v.resize(longest);
    columns.resize(size);

    // Проход по столбцам: квадрат расстояния до ближайшей занятой клетки своего столбца.
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) f[static_cast<std::size_t>(y)] = occupied[static_cast<std::size_t>(y) * width + x] ? 0.0 : kFar;
        transform_1d(f.data(), height, d.data(), v.data(), z.data());
        for (int y = 0; y < height; ++y) columns[static_cast<std::size_t>(y) * width + x] = d[static_cast<std::size_t>(y)];
    }
    // Проход по строкам объединяет столбцы: итог — квадрат евклидова расстояния.
    for (int y = 0; y < height; ++y) {
        const double* row = &columns[static_cast<std::size_t>(y) * width];
        transform_1d(row, width, d.data(), v.data(), z.data());
        for (int x = 0; x < width; ++x) {
            distance[static_cast<std::size_t>(y) * width + x] = static_cast<float>(std::sqrt(d[static_cast<std::size_t>(x)]));
        }
    }
}

GridPlanner::GridPlanner(const PlannerConfig& config) : cfg_(config) {
    if (!(cfg_.robot_radius > 0.0) || cfg_.safety_margin < 0.0) {
        throw std::invalid_argument("PlannerConfig: robot_radius must be > 0 and safety_margin >= 0");
    }
    if (!(cfg_.soft_radius > cfg_.robot_radius + cfg_.safety_margin)) {
        throw std::invalid_argument("PlannerConfig: soft_radius must exceed robot_radius + safety_margin");
    }
}

void GridPlanner::set_map(const OccupancyGrid& map) {
    cells_.resize(static_cast<std::size_t>(map.width()) * map.height());
    for (int iy = 0; iy < map.height(); ++iy) {
        for (int ix = 0; ix < map.width(); ++ix) cells_[static_cast<std::size_t>(iy) * map.width() + ix] = map.state(ix, iy);
    }
    width_ = map.width();
    height_ = map.height();
    resolution_ = map.resolution();
    origin_ = map.origin();
    rebuild_cost(cells_);
}

void GridPlanner::set_map(const OccupancyGrid& map, Vec2 a, Vec2 b) {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!map.observed_bounds(x0, y0, x1, y1)) {
        set_map(map);
        return;
    }
    const auto include = [&](Vec2 p) {
        int ix = 0, iy = 0;
        map.cell_of(p, ix, iy);
        x0 = std::min(x0, ix);
        x1 = std::max(x1, ix);
        y0 = std::min(y0, iy);
        y1 = std::max(y1, iy);
    };
    include(a);
    include(b);
    // Запас по краям: путь в обход препятствия на границе разведанного должен
    // иметь возможность пройти по неразведанной территории рядом.
    const int pad = static_cast<int>(std::ceil(1.0 / map.resolution()));
    x0 = std::max(0, x0 - pad);
    y0 = std::max(0, y0 - pad);
    x1 = std::min(map.width() - 1, x1 + pad);
    y1 = std::min(map.height() - 1, y1 + pad);
    if (x1 < x0 || y1 < y0) {
        set_map(map);
        return;
    }
    width_ = x1 - x0 + 1;
    height_ = y1 - y0 + 1;
    resolution_ = map.resolution();
    origin_ = {map.origin().x + x0 * resolution_, map.origin().y + y0 * resolution_};
    cells_.resize(static_cast<std::size_t>(width_) * height_);
    for (int iy = 0; iy < height_; ++iy) {
        for (int ix = 0; ix < width_; ++ix) cells_[static_cast<std::size_t>(iy) * width_ + ix] = map.state(x0 + ix, y0 + iy);
    }
    rebuild_cost(cells_);
}

void GridPlanner::set_map(const World& world) {
    width_ = world.width();
    height_ = world.height();
    resolution_ = world.cell_size();
    origin_ = world.origin();
    cells_.resize(static_cast<std::size_t>(width_) * height_);
    const std::span<const std::uint8_t> grid = world.cells();
    for (std::size_t i = 0; i < cells_.size(); ++i) cells_[i] = grid[i] != 0 ? CellState::Occupied : CellState::Free;
    // Аналитические препятствия растеризуются по центрам клеток.
    const auto mark = [&](Vec2 lo, Vec2 hi, const auto& inside) {
        const int x0 = std::max(0, static_cast<int>(std::floor((lo.x - origin_.x) / resolution_)));
        const int y0 = std::max(0, static_cast<int>(std::floor((lo.y - origin_.y) / resolution_)));
        const int x1 = std::min(width_ - 1, static_cast<int>(std::floor((hi.x - origin_.x) / resolution_)));
        const int y1 = std::min(height_ - 1, static_cast<int>(std::floor((hi.y - origin_.y) / resolution_)));
        for (int iy = y0; iy <= y1; ++iy) {
            for (int ix = x0; ix <= x1; ++ix) {
                if (inside(world.cell_center(ix, iy))) cells_[static_cast<std::size_t>(iy) * width_ + ix] = CellState::Occupied;
            }
        }
    };
    for (const Aabb& box : world.boxes()) {
        mark(box.min, box.max, [&](Vec2 p) { return p.x >= box.min.x && p.x <= box.max.x && p.y >= box.min.y && p.y <= box.max.y; });
    }
    for (const Circle& circle : world.circles()) {
        const Vec2 r{circle.radius, circle.radius};
        mark(circle.center - r, circle.center + r, [&](Vec2 p) { return distance(p, circle.center) <= circle.radius; });
    }
    rebuild_cost(cells_);
}

void GridPlanner::set_cells(std::span<const CellState> cells, int width, int height, double resolution, Vec2 origin) {
    if (width <= 0 || height <= 0 || !(resolution > 0.0) || cells.size() != static_cast<std::size_t>(width) * height) {
        throw std::invalid_argument("GridPlanner::set_cells: bad grid description");
    }
    width_ = width;
    height_ = height;
    resolution_ = resolution;
    origin_ = origin;
    cells_.assign(cells.begin(), cells.end());
    rebuild_cost(cells_);
}

void GridPlanner::rebuild_cost(std::span<const CellState> cells) {
    const std::size_t size = cells.size();
    occupied_.resize(size);
    for (std::size_t i = 0; i < size; ++i) occupied_[i] = cells[i] == CellState::Occupied ? 1 : 0;
    distance_transform(occupied_, width_, height_, clearance_);

    cost_.resize(size);
    const double lethal = cfg_.robot_radius + cfg_.safety_margin;
    const double soft_span = cfg_.soft_radius - lethal;
    for (std::size_t i = 0; i < size; ++i) {
        // Преобразование даёт расстояние между центрами клеток; до границы занятой
        // клетки на полклетки ближе.
        const double clear = std::max(0.0, std::min(1e3, clearance_[i] * resolution_) - 0.5 * resolution_);
        clearance_[i] = static_cast<float>(clear);
        const bool unknown = cells[i] == CellState::Unknown;
        if (occupied_[i] != 0 || clear < lethal || (unknown && !cfg_.allow_unknown)) {
            cost_[i] = 0.0F;
            continue;
        }
        const double closeness = std::clamp((cfg_.soft_radius - clear) / soft_span, 0.0, 1.0);
        cost_[i] = static_cast<float>(1.0 + cfg_.soft_weight * closeness * closeness + (unknown ? cfg_.unknown_penalty : 0.0));
    }

    // Геометрия сетки могла измениться (окно растёт вместе с разведанной
    // областью), поэтому отметки прошлых поисков сбрасываются.
    g_.assign(size, 0.0F);
    parent_.assign(size, -1);
    seen_.assign(size, 0);
    closed_.assign(size, 0);
    generation_ = 0;
}

int GridPlanner::cell_index(Vec2 p) const noexcept {
    const int ix = static_cast<int>(std::floor((p.x - origin_.x) / resolution_));
    const int iy = static_cast<int>(std::floor((p.y - origin_.y) / resolution_));
    if (ix < 0 || iy < 0 || ix >= width_ || iy >= height_) return -1;
    return iy * width_ + ix;
}

double GridPlanner::clearance(Vec2 p) const noexcept {
    const int cell = width_ > 0 ? cell_index(p) : -1;
    return cell < 0 ? 1e3 : clearance_[static_cast<std::size_t>(cell)];
}

int GridPlanner::nearest_traversable(int cell) const noexcept {
    if (cell < 0) return -1;
    if (cost_[static_cast<std::size_t>(cell)] != 0.0F) return cell;
    constexpr int kRadius = 10;
    const int cx = cell % width_, cy = cell / width_;
    int best = -1, best_d2 = (kRadius + 1) * (kRadius + 1) * 2;
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= width_ || y >= height_) continue;
            const int d2 = dx * dx + dy * dy;
            if (d2 < best_d2 && cost_[static_cast<std::size_t>(y) * width_ + x] != 0.0F) {
                best_d2 = d2;
                best = y * width_ + x;
            }
        }
    }
    return best;
}

// Прямая видимость между клетками (линия Брезенхема). Одновременно накапливается
// средний множитель стоимости вдоль отрезка — длина прямого перехода Theta*
// учитывает близость стен так же, как сеточные шаги.
bool GridPlanner::line_of_sight(int from, int to, double* cost) const noexcept {
    int x = from % width_, y = from / width_;
    const int x1 = to % width_, y1 = to / width_;
    const int dx = std::abs(x1 - x), sx = x < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y), sy = y < y1 ? 1 : -1;
    const double length = std::sqrt(static_cast<double>(dx) * dx + static_cast<double>(dy) * dy) * resolution_;
    int err = dx + dy;
    double sum = 0.0;
    int count = 0;
    for (;;) {
        const float c = cost_[static_cast<std::size_t>(y) * width_ + x];
        if (c == 0.0F) return false;
        sum += c;
        ++count;
        if (x == x1 && y == y1) break;
        const int e2 = 2 * err;
        const bool step_x = e2 >= dy, step_y = e2 <= dx;
        if (step_x && step_y) {
            // Диагональный шаг: нельзя протискиваться между двумя запретными клетками.
            if (cost_[static_cast<std::size_t>(y) * width_ + (x + sx)] == 0.0F ||
                cost_[static_cast<std::size_t>(y + sy) * width_ + x] == 0.0F) {
                return false;
            }
        }
        if (step_x) {
            err += dy;
            x += sx;
        }
        if (step_y) {
            err += dx;
            y += sy;
        }
    }
    if (cost != nullptr) *cost = length * sum / count;
    return true;
}

bool GridPlanner::line_free(Vec2 a, Vec2 b) const noexcept {
    if (width_ == 0) return true;
    const int from = cell_index(a), to = cell_index(b);
    if (from < 0 || to < 0) return false;
    return line_of_sight(from, to, nullptr);
}

bool GridPlanner::plan(Vec2 start, Vec2 goal, std::vector<Vec2>& path) {
    path.clear();
    expansions_ = 0;
    path_cost_ = 0.0;
    if (width_ == 0) return false;
    const int start_cell = cell_index(start), goal_cell = cell_index(goal);
    const int source = nearest_traversable(start_cell), target = nearest_traversable(goal_cell);
    if (source < 0 || target < 0) return false;

    if (++generation_ == 0) {  // переполнение счётчика поколений: начинаем заново
        std::fill(seen_.begin(), seen_.end(), 0U);
        std::fill(closed_.begin(), closed_.end(), 0U);
        generation_ = 1;
    }
    const bool any_angle = cfg_.algorithm == PlannerConfig::Algorithm::ThetaStar;
    const int tx = target % width_, ty = target / width_;
    // Эвристика — нижняя оценка оставшегося пути: множитель стоимости не меньше 1.
    // Для Theta* это расстояние по прямой, для A* — октильное (точнее для сетки).
    const auto heuristic = [&](int cell) {
        const double dx = std::abs(cell % width_ - tx), dy = std::abs(cell / width_ - ty);
        const double cells = any_angle ? std::sqrt(dx * dx + dy * dy)
                                       : std::max(dx, dy) + (std::numbers::sqrt2 - 1.0) * std::min(dx, dy);
        return static_cast<float>(cells * resolution_);
    };

    using Entry = std::pair<float, std::int32_t>;  // (f = g + h, клетка)
    std::vector<Entry> heap;
    const auto push = [&](float f, int cell) {
        heap.emplace_back(f, cell);
        std::push_heap(heap.begin(), heap.end(), std::greater<>{});
    };
    const auto src = static_cast<std::size_t>(source);
    g_[src] = 0.0F;
    parent_[src] = source;
    seen_[src] = generation_;
    push(heuristic(source), source);

    bool found = false;
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<>{});
        const int node = heap.back().second;
        heap.pop_back();
        const auto ni = static_cast<std::size_t>(node);
        if (closed_[ni] == generation_) continue;  // устаревшая запись в куче
        closed_[ni] = generation_;
        if (node == target) {
            found = true;
            break;
        }
        if (++expansions_ > static_cast<std::size_t>(cfg_.max_expansions)) break;

        const int cx = node % width_, cy = node / width_;
        const int grand = parent_[ni];
        for (int k = 0; k < 8; ++k) {
            const int x = cx + kDx[k], y = cy + kDy[k];
            if (x < 0 || y < 0 || x >= width_ || y >= height_) continue;
            const int next = y * width_ + x;
            const auto xi = static_cast<std::size_t>(next);
            if (cost_[xi] == 0.0F || closed_[xi] == generation_) continue;
            if (k >= 4 && (cost_[static_cast<std::size_t>(cy) * width_ + x] == 0.0F ||
                           cost_[static_cast<std::size_t>(y) * width_ + cx] == 0.0F)) {
                continue;  // диагональ срезала бы угол запретной зоны
            }
            // Theta*: если «дедушка» видит соседа напрямую, идём к соседу от него,
            // минуя текущую клетку, — так путь спрямляется прямо во время поиска.
            double candidate = 0.0;
            int via = node;
            double line_cost = 0.0;
            if (any_angle && grand != node && line_of_sight(grand, next, &line_cost)) {
                candidate = g_[static_cast<std::size_t>(grand)] + line_cost;
                via = grand;
            } else {
                const double step = (k < 4 ? 1.0 : std::numbers::sqrt2) * resolution_;
                candidate = g_[ni] + step * 0.5 * (cost_[ni] + cost_[xi]);
            }
            if (seen_[xi] != generation_ || candidate < g_[xi]) {
                seen_[xi] = generation_;
                g_[xi] = static_cast<float>(candidate);
                parent_[xi] = via;
                push(static_cast<float>(candidate) + heuristic(next), next);
            }
        }
    }
    if (!found) return false;
    path_cost_ = g_[static_cast<std::size_t>(target)];

    // Восстановление пути по ссылкам на родителей: от цели к старту.
    std::vector<int> chain;
    for (int cell = target;; cell = parent_[static_cast<std::size_t>(cell)]) {
        chain.push_back(cell);
        if (cell == source) break;
    }
    std::reverse(chain.begin(), chain.end());

    // A* даёт «лесенку» по клеткам. Спрямление: от текущей вершины ищем самую
    // дальнюю, которая видна напрямую, и переходим сразу к ней.
    if (!any_angle && cfg_.smooth && chain.size() > 2) {
        std::vector<int> pulled{chain.front()};
        std::size_t anchor = 0;
        while (anchor + 1 < chain.size()) {
            std::size_t next = anchor + 1;
            for (std::size_t probe = chain.size() - 1; probe > anchor + 1; --probe) {
                if (line_of_sight(chain[anchor], chain[probe], nullptr)) {
                    next = probe;
                    break;
                }
            }
            pulled.push_back(chain[next]);
            anchor = next;
        }
        chain.swap(pulled);
    }

    // Концы пути — точные старт и цель, а не центры клеток.
    path.reserve(chain.size() + 2);
    path.push_back(start);
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const bool is_start_cell = i == 0 && chain[i] == start_cell;
        const bool is_goal_cell = i + 1 == chain.size() && chain[i] == goal_cell;
        if (!is_start_cell && !is_goal_cell) path.push_back(cell_center(chain[i]));
    }
    path.push_back(goal);
    return true;
}

}  // namespace roversim::nav
