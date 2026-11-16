#include "roversim/navgrid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace roversim {
namespace {

constexpr float kUnreachable = std::numeric_limits<float>::infinity();
constexpr int kSearchRadius = 3;  // радиус поиска ближайшей проходимой клетки, в клетках

// Восемь соседей: сначала четыре прямых, затем четыре диагональных.
constexpr int kDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
constexpr int kDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

}  // namespace

NavGrid::NavGrid(const World& world, double radius)
    : width_(world.width()), height_(world.height()), cell_(world.cell_size()), origin_(world.origin()) {
    const std::size_t size = static_cast<std::size_t>(width_) * height_;
    std::vector<std::uint8_t> passable(size, 0);
    for (int iy = 0; iy < height_; ++iy) {
        for (int ix = 0; ix < width_; ++ix) {
            passable[static_cast<std::size_t>(iy) * width_ + ix] = world.is_free(world.cell_center(ix, iy), radius);
        }
    }

    // Разметка 4-связных компонент обходом в ширину; запоминаем наибольшую.
    std::vector<std::int32_t> label(size, -1);
    std::vector<std::int32_t> queue;
    std::int32_t best_label = -1;
    std::size_t best_size = 0;
    std::int32_t next_label = 0;
    for (std::size_t seed = 0; seed < size; ++seed) {
        if (passable[seed] == 0 || label[seed] >= 0) continue;
        queue.clear();
        queue.push_back(static_cast<std::int32_t>(seed));
        label[seed] = next_label;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const int cur = queue[head];
            const int cx = cur % width_, cy = cur / width_;
            for (int k = 0; k < 4; ++k) {
                const int nx = cx + kDx[k], ny = cy + kDy[k];
                if (nx < 0 || ny < 0 || nx >= width_ || ny >= height_) continue;
                const std::size_t ni = static_cast<std::size_t>(ny) * width_ + nx;
                if (passable[ni] == 0 || label[ni] >= 0) continue;
                label[ni] = next_label;
                queue.push_back(static_cast<std::int32_t>(ni));
            }
        }
        if (queue.size() > best_size) {
            best_size = queue.size();
            best_label = next_label;
        }
        ++next_label;
    }

    free_.assign(size, 0);
    free_cells_.reserve(best_size);
    for (std::size_t i = 0; i < size; ++i) {
        if (best_label >= 0 && label[i] == best_label) {
            free_[i] = 1;
            free_cells_.push_back(static_cast<std::int32_t>(i));
        }
    }
}

Vec2 NavGrid::sample(Rng& rng) const noexcept {
    if (free_cells_.empty()) return {};
    return center(free_cells_[rng.below(free_cells_.size())]);
}

int NavGrid::nearest_free(Vec2 p) const noexcept {
    const int ix = static_cast<int>(std::floor((p.x - origin_.x) / cell_));
    const int iy = static_cast<int>(std::floor((p.y - origin_.y) / cell_));
    // Обычный случай: точка уже в проходимой клетке.
    if (ix >= 0 && iy >= 0 && ix < width_ && iy < height_ && free_[static_cast<std::size_t>(iy) * width_ + ix] != 0) {
        return iy * width_ + ix;
    }
    int best = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int dy = -kSearchRadius; dy <= kSearchRadius; ++dy) {
        for (int dx = -kSearchRadius; dx <= kSearchRadius; ++dx) {
            const int nx = ix + dx, ny = iy + dy;
            if (nx < 0 || ny < 0 || nx >= width_ || ny >= height_) continue;
            const int ni = ny * width_ + nx;
            if (free_[static_cast<std::size_t>(ni)] == 0) continue;
            const Vec2 delta = center(ni) - p;
            const double d2 = dot(delta, delta);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = ni;
            }
        }
    }
    return best;
}

void NavGrid::distance_field(Vec2 goal, std::vector<float>& field, std::vector<std::int32_t>& next) const {
    field.assign(free_.size(), kUnreachable);
    next.assign(free_.size(), -1);
    const int goal_cell = nearest_free(goal);
    if (goal_cell < 0) return;

    // Алгоритм Дейкстры с «корзинами» вместо кучи (алгоритм Дайела). Расстояния
    // считаются в клетках; корзина b хранит клетки с расстоянием из [b, b + 1).
    // Любое ребро не короче одной клетки, поэтому релаксация из корзины b попадает
    // только в корзины с большим номером: к моменту обработки корзины все её
    // клетки окончательны, и порядок внутри неё не важен. Сложность линейная.
    // Буферы thread_local: поле строится при каждом reset(), а NavGrid разделяется
    // между потоками пакетного симулятора.
    thread_local std::vector<std::vector<std::int32_t>> buckets;
    for (auto& bucket : buckets) bucket.clear();
    if (buckets.empty()) buckets.resize(64);

    constexpr float kStraight = 1.0F;
    const float diagonal = static_cast<float>(std::numbers::sqrt2);
    field[static_cast<std::size_t>(goal_cell)] = 0.0F;
    buckets[0].push_back(goal_cell);
    for (std::size_t b = 0; b < buckets.size(); ++b) {
        // Обращение только по индексу: buckets может перераспределиться при вставке.
        for (std::size_t k = 0; k < buckets[b].size(); ++k) {
            const int cur = buckets[b][k];
            const float dist = field[static_cast<std::size_t>(cur)];
            if (static_cast<std::size_t>(dist) != b) continue;  // запись устарела: найден путь короче
            const int cx = cur % width_, cy = cur / width_;
            for (int n = 0; n < 8; ++n) {
                const int nx = cx + kDx[n], ny = cy + kDy[n];
                if (nx < 0 || ny < 0 || nx >= width_ || ny >= height_) continue;
                const int ni = ny * width_ + nx;
                if (free_[static_cast<std::size_t>(ni)] == 0) continue;
                // По диагонали можно идти, только если проходимы обе смежные клетки,
                // иначе путь срезал бы угол препятствия.
                if (n >= 4 && (free_[static_cast<std::size_t>(cy * width_ + nx)] == 0 ||
                               free_[static_cast<std::size_t>(ny * width_ + cx)] == 0)) {
                    continue;
                }
                const float candidate = dist + (n < 4 ? kStraight : diagonal);
                if (candidate < field[static_cast<std::size_t>(ni)]) {
                    field[static_cast<std::size_t>(ni)] = candidate;
                    next[static_cast<std::size_t>(ni)] = cur;  // волна идёт от цели: из ni к цели — через cur
                    const auto target = static_cast<std::size_t>(candidate);
                    if (target >= buckets.size()) buckets.resize(target + 1);
                    buckets[target].push_back(ni);
                }
            }
        }
    }

    const float cell = static_cast<float>(cell_);
    for (float& value : field) value *= cell;  // клетки → метры (inf остаётся inf)
}

double NavGrid::lookup(const std::vector<float>& field, Vec2 p) const noexcept {
    const double unreachable = std::numeric_limits<double>::infinity();
    if (field.size() != free_.size()) return unreachable;

    // Поле задано в центрах клеток. Между ними оно интерполируется билинейно:
    // расстояние меняется непрерывно, и награда за прогресс получается плавной,
    // а не «ноль, ноль, скачок на размер клетки».
    const double gx = (p.x - origin_.x) / cell_ - 0.5;
    const double gy = (p.y - origin_.y) / cell_ - 0.5;
    const int ix = static_cast<int>(std::floor(gx)), iy = static_cast<int>(std::floor(gy));
    const double fx = gx - ix, fy = gy - iy;
    double corner[4];
    int finite = 0;
    for (int k = 0; k < 4; ++k) {
        const int nx = ix + (k & 1), ny = iy + (k >> 1);
        corner[k] = unreachable;
        if (nx < 0 || ny < 0 || nx >= width_ || ny >= height_) continue;
        const float value = field[static_cast<std::size_t>(ny * width_ + nx)];
        if (value < kUnreachable) {
            corner[k] = value;
            ++finite;
        }
    }
    if (finite == 4) {
        const double bottom = corner[0] + (corner[1] - corner[0]) * fx;
        const double top = corner[2] + (corner[3] - corner[2]) * fx;
        return bottom + (top - bottom) * fy;
    }

    // У стены часть соседних клеток непроходима. Тогда берём лучший маршрут вида
    // «по прямой до центра проходимой клетки, дальше по полю».
    double best = unreachable;
    for (int k = 0; k < 4; ++k) {
        if (corner[k] < unreachable) {
            best = std::min(best, corner[k] + distance(center((iy + (k >> 1)) * width_ + ix + (k & 1)), p));
        }
    }
    if (best < unreachable) return best;
    const int near = nearest_free(p);
    if (near < 0 || !(field[static_cast<std::size_t>(near)] < kUnreachable)) return unreachable;
    return field[static_cast<std::size_t>(near)] + distance(center(near), p);
}

bool NavGrid::line_free(Vec2 a, Vec2 b) const noexcept {
    // Проход по отрезку с шагом в полклетки: проверяется каждая клетка под ним.
    const double length = distance(a, b);
    const int steps = std::max(1, static_cast<int>(std::ceil(length / (0.5 * cell_))));
    const double inv_cell = 1.0 / cell_;
    for (int k = 0; k <= steps; ++k) {
        const Vec2 p = a + (b - a) * (static_cast<double>(k) / steps);
        const int ix = static_cast<int>(std::floor((p.x - origin_.x) * inv_cell));
        const int iy = static_cast<int>(std::floor((p.y - origin_.y) * inv_cell));
        if (ix < 0 || iy < 0 || ix >= width_ || iy >= height_) return false;
        if (free_[static_cast<std::size_t>(iy) * width_ + ix] == 0) return false;
    }
    return true;
}

Vec2 NavGrid::lookahead(const std::vector<std::int32_t>& next, Vec2 start, Vec2 goal, double ahead) const noexcept {
    const auto toward = [&](Vec2 target) {  // точка на луче к target, не дальше ahead
        const double d = distance(start, target);
        return d <= ahead ? target : start + (target - start) * (ahead / d);
    };
    // Цель видна по прямой — правим прямо на неё: это и есть кратчайший путь.
    if (next.size() != free_.size() || line_free(start, goal)) return toward(goal);
    int cur = nearest_free(start);
    if (cur < 0) return toward(goal);

    // Клетки кратчайшего пути в пределах нескольких ahead — по ссылкам next.
    constexpr int kMaxCells = 96;
    int cells[kMaxCells];
    int count = 0;
    const double horizon = 3.0 * ahead;
    for (double walked = 0.0; count < kMaxCells && walked < horizon;) {
        const int following = next[static_cast<std::size_t>(cur)];
        if (following < 0) break;  // цель или тупик
        walked += distance(center(cur), center(following));
        cur = following;
        cells[count++] = cur;
    }
    if (count == 0) return toward(goal);

    // Самая дальняя точка пути, видимая из start по прямой. Вдоль кратчайшего
    // пути видимость пропадает один раз — за углом, — поэтому граница ищется
    // двоичным поиском: несколько проверок вместо проверки каждой клетки.
    int visible = 0, hidden = count - 1;
    if (line_free(start, center(cells[hidden]))) {
        visible = hidden;
    } else {
        while (hidden - visible > 1) {
            const int mid = (visible + hidden) / 2;
            if (line_free(start, center(cells[mid]))) {
                visible = mid;
            } else {
                hidden = mid;
            }
        }
    }
    return toward(center(cells[visible]));
}

std::vector<Vec2> NavGrid::path(const std::vector<std::int32_t>& next, Vec2 start, Vec2 goal) const {
    std::vector<Vec2> points;
    if (next.size() != free_.size()) return points;
    int cur = nearest_free(start);
    if (cur < 0) return points;
    // Недостижимая клетка не имеет ссылки и при этом не является клеткой цели.
    const int goal_cell = nearest_free(goal);
    if (next[static_cast<std::size_t>(cur)] < 0 && cur != goal_cell) return points;

    points.push_back(start);
    for (int following = next[static_cast<std::size_t>(cur)]; following >= 0; following = next[static_cast<std::size_t>(cur)]) {
        cur = following;
        points.push_back(center(cur));
    }
    points.push_back(goal);
    return points;
}

}  // namespace roversim
