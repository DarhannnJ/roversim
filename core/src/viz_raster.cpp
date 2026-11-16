#include "roversim/viz.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace roversim::viz {
namespace {

struct Color {
    std::uint8_t r, g, b;
};

constexpr Color kOutside{11, 13, 17};
constexpr Color kFloor{24, 27, 34};
constexpr Color kGridLine{32, 36, 45};
constexpr Color kWall{196, 202, 212};
constexpr Color kRay{64, 200, 120};
constexpr Color kHit{255, 96, 80};
constexpr Color kTrail{120, 160, 255};
constexpr Color kGoal{250, 200, 40};
constexpr Color kOdom{255, 140, 0};
constexpr Color kObstacle{186, 104, 200};
constexpr Color kRobot{66, 135, 245};
constexpr Color kRobotHit{235, 70, 60};
constexpr Color kHeading{255, 255, 255};

constexpr double kPadding = 0.04;  // поля кадра, доля от меньшей стороны

/// Примитивы рисования с альфа-смешиванием. Координаты — пиксели, y вниз.
class Canvas {
 public:
    explicit Canvas(Image& image) : image_(image) {}

    void blend(int x, int y, Color c, double alpha = 1.0) {
        if (x < 0 || y < 0 || x >= image_.width || y >= image_.height) return;
        std::uint8_t* p = &image_.rgb[(static_cast<std::size_t>(y) * image_.width + x) * 3];
        p[0] = mix(p[0], c.r, alpha);
        p[1] = mix(p[1], c.g, alpha);
        p[2] = mix(p[2], c.b, alpha);
    }

    void line(Vec2 a, Vec2 b, Color c, double alpha = 1.0) {
        const double dx = b.x - a.x, dy = b.y - a.y;
        const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)))));
        for (int i = 0; i <= steps; ++i) {
            const double t = static_cast<double>(i) / steps;
            blend(static_cast<int>(std::floor(a.x + dx * t)), static_cast<int>(std::floor(a.y + dy * t)), c, alpha);
        }
    }

    /// Закрашенный круг при thickness <= 0, иначе кольцо заданной толщины.
    void circle(Vec2 center, double radius, Color c, double alpha = 1.0, double thickness = 0.0) {
        const double outer = radius + std::max(thickness, 0.0) * 0.5;
        const double inner = thickness > 0.0 ? radius - thickness * 0.5 : -1.0;
        const int x0 = static_cast<int>(std::floor(center.x - outer)), x1 = static_cast<int>(std::ceil(center.x + outer));
        const int y0 = static_cast<int>(std::floor(center.y - outer)), y1 = static_cast<int>(std::ceil(center.y + outer));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double d = std::hypot(x + 0.5 - center.x, y + 0.5 - center.y);
                if (d <= outer && d >= inner) blend(x, y, c, alpha);
            }
        }
    }

    void rect(Vec2 a, Vec2 b, Color c) {
        const int x0 = static_cast<int>(std::floor(std::min(a.x, b.x))), x1 = static_cast<int>(std::ceil(std::max(a.x, b.x)));
        const int y0 = static_cast<int>(std::floor(std::min(a.y, b.y))), y1 = static_cast<int>(std::ceil(std::max(a.y, b.y)));
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) blend(x, y, c);
        }
    }

 private:
    static std::uint8_t mix(std::uint8_t from, std::uint8_t to, double alpha) {
        return static_cast<std::uint8_t>(std::lround(from + (to - from) * alpha));
    }

    Image& image_;
};

}  // namespace

Renderer::Renderer(int width, int height) {
    if (width < 16 || height < 16 || width > 8192 || height > 8192) {
        throw std::invalid_argument("Renderer: image size must be within [16, 8192]");
    }
    frame_.width = background_.width = width;
    frame_.height = background_.height = height;
    frame_.rgb.assign(static_cast<std::size_t>(width) * height * 3, 0);
    background_.rgb = frame_.rgb;
}

void Renderer::rebuild_background(const World& world) {
    world_ = &world;
    world_revision_ = world.revision();
    trail_.clear();

    // Масштаб: мир целиком вписывается в кадр с полями, ось Y мира направлена вверх.
    const Aabb b = world.bounds();
    const int w = background_.width, h = background_.height;
    const double pad = kPadding * std::min(w, h);
    scale_ = std::min((w - 2.0 * pad) / (b.max.x - b.min.x), (h - 2.0 * pad) / (b.max.y - b.min.y));
    offset_x_ = 0.5 * (w - (b.max.x - b.min.x) * scale_) - b.min.x * scale_;
    offset_y_ = 0.5 * (h + (b.max.y - b.min.y) * scale_) + b.min.y * scale_;

    const double cell = world.cell_size();
    const double half_pixel = 0.5 / scale_;
    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            const double x = (px + 0.5 - offset_x_) / scale_;
            const double y = (offset_y_ - (py + 0.5)) / scale_;
            Color c = kOutside;
            if (x >= b.min.x && x < b.max.x && y >= b.min.y && y < b.max.y) {
                const int ix = static_cast<int>((x - b.min.x) / cell), iy = static_cast<int>((y - b.min.y) / cell);
                if (world.occupied(ix, iy)) {
                    c = kWall;
                } else {
                    // Метровая сетка на полу — для оценки масштаба на глаз.
                    const bool on_line = std::abs(x - std::round(x)) < half_pixel || std::abs(y - std::round(y)) < half_pixel;
                    c = on_line ? kGridLine : kFloor;
                }
            }
            std::uint8_t* p = &background_.rgb[(static_cast<std::size_t>(py) * w + px) * 3];
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
        }
    }

    Canvas canvas(background_);
    const auto to_pixel = [&](Vec2 p) { return Vec2{offset_x_ + p.x * scale_, offset_y_ - p.y * scale_}; };
    for (const Aabb& box : world.boxes()) canvas.rect(to_pixel(box.min), to_pixel(box.max), kWall);
    for (const Circle& circle : world.circles()) canvas.circle(to_pixel(circle.center), circle.radius * scale_, kWall);
}

const Image& Renderer::render(const Simulator& sim) {
    const World& world = sim.world();
    if (&world != world_ || world.revision() != world_revision_) rebuild_background(world);
    frame_.rgb = background_.rgb;

    Canvas canvas(frame_);
    const auto to_pixel = [&](Vec2 p) { return Vec2{offset_x_ + p.x * scale_, offset_y_ - p.y * scale_}; };
    const RobotState& robot = sim.robot_state();
    const Vec2 position = robot.pose.position();
    const double c = std::cos(robot.pose.theta), s = std::sin(robot.pose.theta);
    const double radius = sim.config().robot.radius;

    // След: начинается заново с каждым эпизодом.
    if (sim.episode() != episode_) {
        trail_.clear();
        episode_ = sim.episode();
    }
    if (trail_.empty() || distance(trail_.back(), position) > 0.02) trail_.push_back(position);

    // Скан лидара: лучи и точки отражений.
    const LidarConfig& lidar = sim.config().lidar;
    const Lidar2D& model = sim.lidar_model();
    const Vec2 sensor = position + rotate({lidar.mount.x, lidar.mount.y}, c, s);
    const Vec2 sensor_px = to_pixel(sensor);
    const std::span<const float> ranges = sim.lidar();
    const double yaw = robot.pose.theta + lidar.mount.theta + model.angle_min();
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const double a = yaw + static_cast<double>(i) * model.angle_increment();
        const Vec2 hit_px = to_pixel(sensor + Vec2{std::cos(a), std::sin(a)} * ranges[i]);
        canvas.line(sensor_px, hit_px, kRay, 0.16);
        if (ranges[i] < lidar.range_max - 1e-3) canvas.circle(hit_px, 1.3, kHit);
    }

    for (std::size_t i = 1; i < trail_.size(); ++i) {
        canvas.line(to_pixel(trail_[i - 1]), to_pixel(trail_[i]), kTrail, 0.9);
    }

    // Подвижные препятствия: круг и отрезок в направлении движения (путь за 0.5 с).
    for (const DynamicObstacle& obstacle : sim.obstacles()) {
        const Vec2 center_px = to_pixel(obstacle.position);
        canvas.circle(center_px, obstacle.radius * scale_, kObstacle, 0.95);
        canvas.line(center_px, to_pixel(obstacle.position + obstacle.velocity * 0.5), kHeading, 0.8);
    }

    // Цель: кольцо радиуса достижения и точка в центре.
    const Vec2 goal_px = to_pixel(sim.goal());
    canvas.circle(goal_px, sim.config().task.goal_radius * scale_, kGoal, 0.9, 2.0);
    canvas.circle(goal_px, 2.5, kGoal);

    // Счисленная по одометрии поза — контур: наглядно виден дрейф.
    const Pose2& odom = sim.odom_pose();
    canvas.circle(to_pixel(odom.position()), radius * scale_, kOdom, 0.9, 1.5);

    // Робот: красный, пока касается препятствия.
    const Vec2 robot_px = to_pixel(position);
    canvas.circle(robot_px, radius * scale_, sim.last_result().collision ? kRobotHit : kRobot, 0.95);
    canvas.line(robot_px, to_pixel(position + Vec2{c, s} * radius), kHeading);
    return frame_;
}

std::string to_ansi(const Image& image) {
    std::string out;
    out.reserve(static_cast<std::size_t>(image.width) * image.height * 4);
    char code[40];
    for (int y = 0; y + 1 < image.height; y += 2) {
        // Цвет задаётся заново только при смене: ровные области сжимаются в разы.
        int last_top = -1, last_bottom = -1;
        for (int x = 0; x < image.width; ++x) {
            const std::uint8_t* t = &image.rgb[(static_cast<std::size_t>(y) * image.width + x) * 3];
            const std::uint8_t* b = t + static_cast<std::size_t>(image.width) * 3;
            const int top = (t[0] << 16) | (t[1] << 8) | t[2];
            const int bottom = (b[0] << 16) | (b[1] << 8) | b[2];
            if (top != last_top) {
                out.append(code, static_cast<std::size_t>(std::snprintf(code, sizeof(code), "\x1b[38;2;%d;%d;%dm", t[0], t[1], t[2])));
                last_top = top;
            }
            if (bottom != last_bottom) {
                out.append(code, static_cast<std::size_t>(std::snprintf(code, sizeof(code), "\x1b[48;2;%d;%d;%dm", b[0], b[1], b[2])));
                last_bottom = bottom;
            }
            out += "\xE2\x96\x80";  // «▀»: верхняя половина — цвет символа, нижняя — цвет фона
        }
        out += "\x1b[0m\n";
    }
    return out;
}

bool write_ppm(const Image& image, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.rgb.data()), static_cast<std::streamsize>(image.rgb.size()));
    return static_cast<bool>(file);
}

}  // namespace roversim::viz
