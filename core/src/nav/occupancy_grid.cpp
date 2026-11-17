#include "roversim/nav/occupancy_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace roversim::nav {

OccupancyGrid::OccupancyGrid(const MapConfig& config, Vec2 center) : cfg_(config) {
    if (!(cfg_.resolution > 0.0) || !(cfg_.size_x > 0.0) || !(cfg_.size_y > 0.0)) {
        throw std::invalid_argument("MapConfig: resolution and sizes must be positive");
    }
    if (!(cfg_.log_odds_hit > 0.0) || !(cfg_.log_odds_miss > 0.0) || !(cfg_.log_odds_min < 0.0) ||
        !(cfg_.log_odds_max > 0.0)) {
        throw std::invalid_argument("MapConfig: log-odds increments and limits have wrong signs");
    }
    width_ = static_cast<int>(std::ceil(cfg_.size_x / cfg_.resolution));
    height_ = static_cast<int>(std::ceil(cfg_.size_y / cfg_.resolution));
    if (static_cast<long long>(width_) * height_ > (1LL << 26)) throw std::invalid_argument("MapConfig: map is too large");
    origin_ = {center.x - 0.5 * width_ * cfg_.resolution, center.y - 0.5 * height_ * cfg_.resolution};
    const auto size = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    log_odds_.assign(size, 0.0F);
    floor_count_.assign(size, 0);
    hit_count_.assign(size, 0);
    mark_.assign(size, 0);
}

bool OccupancyGrid::cell_of(Vec2 p, int& ix, int& iy) const noexcept {
    ix = static_cast<int>(std::floor((p.x - origin_.x) / cfg_.resolution));
    iy = static_cast<int>(std::floor((p.y - origin_.y) / cfg_.resolution));
    return ix >= 0 && iy >= 0 && ix < width_ && iy < height_;
}

double OccupancyGrid::probability(int ix, int iy) const noexcept {
    return 1.0 / (1.0 + std::exp(-static_cast<double>(log_odds(ix, iy))));
}

double OccupancyGrid::occupancy_value(Vec2 p) const noexcept {
    // Координаты в единицах клеток относительно центра клетки (0, 0).
    const double gx = (p.x - origin_.x) / cfg_.resolution - 0.5;
    const double gy = (p.y - origin_.y) / cfg_.resolution - 0.5;
    const double fx0 = std::floor(gx), fy0 = std::floor(gy);
    const int ix = static_cast<int>(fx0), iy = static_cast<int>(fy0);
    const double tx = gx - fx0, ty = gy - fy0;
    const double scale_pos = 1.0 / cfg_.log_odds_max, scale_neg = -1.0 / cfg_.log_odds_min;
    const auto value = [&](int x, int y) {
        const double l = log_odds(x, y);
        return l >= 0.0 ? l * scale_pos : l * scale_neg;
    };
    const double bottom = value(ix, iy) + (value(ix + 1, iy) - value(ix, iy)) * tx;
    const double top = value(ix, iy + 1) + (value(ix + 1, iy + 1) - value(ix, iy + 1)) * tx;
    return bottom + (top - bottom) * ty;
}

void OccupancyGrid::find_transient_hits(const Pose2& sensor, std::span<const Vec2> dirs, std::span<const float> ranges,
                                        double range_max, std::vector<std::uint8_t>& transient) const {
    const std::size_t n = std::min(dirs.size(), ranges.size());
    transient.assign(ranges.size(), 0);
    const double c = std::cos(sensor.theta), s = std::sin(sensor.theta);
    const double no_return = range_max - 1e-3;
    const auto needed = static_cast<std::uint8_t>(std::clamp(cfg_.floor_confirmations, 1, 255));
    for (std::size_t i = 0; i < n; ++i) {
        if (ranges[i] >= no_return) continue;
        const Vec2 dir = rotate(dirs[i], c, s);
        int ix = 0, iy = 0;
        if (!cell_of(sensor.position() + dir * static_cast<double>(ranges[i]), ix, iy)) continue;
        if (ix < 1 || iy < 1 || ix + 1 >= width_ || iy + 1 >= height_) continue;
        // «Полом» должны быть и сама клетка, и все восемь соседних. У стены это
        // условие не выполняется никогда, поэтому шум дальности и погрешность
        // позы, сдвигающие отражение от стены на соседнюю клетку, ложных
        // срабатываний не дают.
        bool floor = true;
        for (int dy = -1; dy <= 1 && floor; ++dy) {
            const std::uint8_t* row = &floor_count_[static_cast<std::size_t>(iy + dy) * width_ + (ix - 1)];
            floor = row[0] >= needed && row[1] >= needed && row[2] >= needed;
        }
        transient[i] = floor ? 1 : 0;
    }
}

namespace {

constexpr std::int64_t kSubcell = 256;  // долей клетки в фиксированной точке

constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

}  // namespace

/// Помечает клетку свободной (не чаще раза за скан). false — клетка вне карты.
bool OccupancyGrid::mark_free(int x, int y) noexcept {
    if (x < 0 || y < 0 || x >= width_ || y >= height_) return false;
    const std::size_t i = static_cast<std::size_t>(y) * width_ + x;
    const std::uint32_t miss_mark = 2 * scan_id_;
    if (mark_[i] < miss_mark) {  // в этом скане клетку ещё не трогали
        mark_[i] = miss_mark;
        // Память попаданий: наблюдение «свободно» сначала гасит одно прежнее попадание.
        if (hit_count_[i] > 0) {
            --hit_count_[i];
            return true;
        }
        log_odds_[i] = std::max(static_cast<float>(cfg_.log_odds_min), log_odds_[i] - static_cast<float>(cfg_.log_odds_miss));
        // Память о свободных наблюдениях: клетка, уверенно свободная несколько
        // сканов подряд, запоминается как «пол».
        if (log_odds_[i] < static_cast<float>(cfg_.free_threshold) && floor_count_[i] < 255) ++floor_count_[i];
    }
    return true;
}

// Алгоритм Брезенхема: клетки вдоль отрезка от (x0, y0) до (x1, y1). Клетка конца
// не включается: по основной оси обход останавливается перед её столбцом.
// Координаты заданы в клетках с дробной частью.
//
// Идея алгоритма: идти по одной клетке вдоль «длинной» оси и копить в целочисленном
// накопителе, насколько идеальная линия ушла по «короткой» оси; когда накопилось на
// целую клетку — сделать шаг и по ней. Классический вариант начинает с нулевой
// ошибки, то есть проводит линию между центрами клеток. Здесь начальное значение
// накопителя вычисляется по настоящим концам отрезка, поэтому в каждом столбце
// выбирается именно та клетка, через которую луч проходит в его середине.
void OccupancyGrid::trace_free(double x0, double y0, double x1, double y1) noexcept {
    const auto fixed = [](double v) { return static_cast<std::int64_t>(std::floor(v * static_cast<double>(kSubcell))); };
    const std::int64_t X0 = fixed(x0), Y0 = fixed(y0), X1 = fixed(x1), Y1 = fixed(y1);
    const int start_x = static_cast<int>(floor_div(X0, kSubcell)), start_y = static_cast<int>(floor_div(Y0, kSubcell));
    const int end_x = static_cast<int>(floor_div(X1, kSubcell)), end_y = static_cast<int>(floor_div(Y1, kSubcell));
    mark_free(start_x, start_y);
    if (start_x == end_x && start_y == end_y) return;  // свободный участок не выходит из клетки сенсора

    const std::int64_t DX = X1 - X0, DY = Y1 - Y0;
    const std::int64_t adx = DX < 0 ? -DX : DX, ady = DY < 0 ? -DY : DY;
    // Обход вдоль оси `major`; `minor` — поперечная. Лямбда одна на оба случая:
    // оси просто меняются местами.
    const auto walk = [&](std::int64_t M0, std::int64_t N0, std::int64_t dM, std::int64_t dN, std::int64_t adM, int major,
                          int major_end, bool major_is_x) {
        const int step = dM > 0 ? 1 : -1;
        const std::int64_t denom = adM * kSubcell;  // «целая клетка» в единицах накопителя
        major += step;
        // Поперечная координата линии в середине первого столбца после стартового.
        const std::int64_t center = static_cast<std::int64_t>(major) * kSubcell + kSubcell / 2;
        const std::int64_t offset = center > M0 ? center - M0 : M0 - center;
        const std::int64_t numerator = N0 * adM + dN * offset;
        std::int64_t minor = floor_div(numerator, denom);
        std::int64_t error = numerator - minor * denom;  // накопитель: 0 <= error < denom
        const std::int64_t increment = dN * kSubcell;     // прирост за один столбец, |increment| <= denom
        // Условие «не дошли до столбца конца» записано через знак разности, а не
        // через неравенство: у очень короткого отрезка, пересекающего границу
        // клетки только по поперечной оси, столбец конца совпадает со стартовым,
        // и обход не должен начинаться вовсе.
        while ((major_end - major) * step > 0) {
            const bool inside = major_is_x ? mark_free(major, static_cast<int>(minor)) : mark_free(static_cast<int>(minor), major);
            if (!inside) return;  // луч начинается внутри прямоугольной карты: выйдя, он не вернётся
            major += step;
            error += increment;
            if (error >= denom) {
                error -= denom;
                ++minor;
            } else if (error < 0) {
                error += denom;
                --minor;
            }
        }
    };
    if (adx >= ady) {
        walk(X0, Y0, DX, DY, adx, start_x, end_x, true);
    } else {
        walk(Y0, X0, DY, DX, ady, start_y, end_y, false);
    }
}

void OccupancyGrid::integrate_scan(const Pose2& sensor, std::span<const Vec2> dirs, std::span<const float> ranges,
                                   double range_max, std::span<const std::uint8_t> ray_flags) {
    int x0 = 0, y0 = 0;
    if (!cell_of(sensor.position(), x0, y0)) return;  // сенсор вне карты
    ++scan_id_;
    if (max_x_ < min_x_) {
        min_x_ = max_x_ = x0;
        min_y_ = max_y_ = y0;
    }
    const std::size_t n = std::min(dirs.size(), ranges.size());
    end_x_.resize(n);
    end_y_.resize(n);

    const double c = std::cos(sensor.theta), s = std::sin(sensor.theta);
    const double inv_res = 1.0 / cfg_.resolution;
    const double no_return = range_max - 1e-3;
    const std::uint32_t hit_mark = 2 * scan_id_ + 1;
    const float upper = static_cast<float>(cfg_.log_odds_max);
    const float hit = static_cast<float>(cfg_.log_odds_hit);
    const auto hit_memory = static_cast<std::uint8_t>(std::clamp(cfg_.hit_memory, 0, 255));
    // Положение сенсора в единицах клеток (с дробной частью).
    const double sensor_x = (sensor.x - origin_.x) * inv_res, sensor_y = (sensor.y - origin_.y) * inv_res;

    // Синус угла между лучом и поверхностью, в которую он упёрся (1 — по нормали,
    // 0 — вскользь). Касательная к поверхности оценивается по соседним точкам
    // скана, если они лежат на той же поверхности (дальности близки).
    const auto incidence_sine = [&](std::size_t i) {
        const double range = ranges[i];
        const auto same_surface = [&](std::size_t j) { return ranges[j] < no_return && std::abs(ranges[j] - range) < 0.3; };
        const std::size_t a = i > 0 && same_surface(i - 1) ? i - 1 : i;
        const std::size_t b = i + 1 < n && same_surface(i + 1) ? i + 1 : i;
        if (a == b) return 1.0;
        const Vec2 tangent = dirs[b] * static_cast<double>(ranges[b]) - dirs[a] * static_cast<double>(ranges[a]);
        const double length = norm(tangent);
        if (length < 1e-9) return 1.0;
        return std::abs(dirs[i].x * tangent.y - dirs[i].y * tangent.x) / length;
    };

    // Проход 1: концы лучей. Клетки с отражением сразу помечаются занятыми.
    for (std::size_t i = 0; i < n; ++i) {
        const double range = ranges[i];
        const bool has_return = range < no_return;
        const double length = (has_return ? range : std::min(range, cfg_.free_range)) * inv_res;
        const Vec2 dir = rotate(dirs[i], c, s);
        const double hit_x = sensor_x + dir.x * length, hit_y = sensor_y + dir.y * length;
        // Свободный участок луча короче самого луча на «хвост» у поверхности.
        const double tail = has_return ? cfg_.surface_clearance / std::max(incidence_sine(i), 0.05) : 0.0;
        const double free_length = std::max(0.0, length - tail);
        end_x_[i] = sensor_x + dir.x * free_length;
        end_y_[i] = sensor_y + dir.y * free_length;
        const int x1 = static_cast<int>(std::floor(hit_x)), y1 = static_cast<int>(std::floor(hit_y));
        min_x_ = std::min(min_x_, std::max(x1, 0));
        max_x_ = std::max(max_x_, std::min(x1, width_ - 1));
        min_y_ = std::min(min_y_, std::max(y1, 0));
        max_y_ = std::max(max_y_, std::min(y1, height_ - 1));
        const std::uint8_t flag = i < ray_flags.size() ? ray_flags[i] : ray_flag::Static;
        if (!has_return || flag == ray_flag::Dynamic || x1 < 0 || y1 < 0 || x1 >= width_ || y1 >= height_) continue;
        const std::size_t cell = static_cast<std::size_t>(y1) * width_ + x1;
        // Объект признан неподвижным: его клетка больше не «пол», и отражения от
        // него перестают выглядеть как появление чего-то нового.
        if (flag == ray_flag::Settled) floor_count_[cell] = 0;
        if (mark_[cell] != hit_mark) {
            mark_[cell] = hit_mark;
            // Запоминаются только попадания, которые подтверждают уже уверенно
            // занятую клетку (оценка на верхней границе). У стены таких сотни; у
            // объекта, проезжающего через клетку за несколько сканов, их почти
            // нет, и его след на карте стирается без задержки.
            const bool certain = log_odds_[cell] >= upper;
            log_odds_[cell] = std::min(upper, log_odds_[cell] + hit);
            if (certain && hit_count_[cell] < hit_memory) ++hit_count_[cell];
        }
    }

    // Проход 2: свободное пространство вдоль каждого луча (до начала «хвоста»).
    // Клетки, помеченные на первом проходе занятыми, пропускаются: их отметка
    // больше отметки «свободно».
    for (std::size_t i = 0; i < n; ++i) trace_free(sensor_x, sensor_y, end_x_[i], end_y_[i]);
}

}  // namespace roversim::nav
