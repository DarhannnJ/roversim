// Встроенный веб-просмотрщик: минимальный HTTP-сервер + WebSocket (RFC 6455).
//
// Один фоновый поток обслуживает все соединения через poll(). Поток симуляции
// только кладёт готовый бинарный кадр в общий слот (publish), а рассылкой
// занимается сервер — поэтому медленный или зависший браузер не тормозит
// симуляцию: он просто получает кадры реже.
//
// Протокол (little-endian), сервер → браузер:
//   тип 1 «мир»:  u8 type, u8[3], u32 w, u32 h, f32 cell, f32 ox, f32 oy,
//                 u32 n_boxes, u32 n_circles, f32 boxes[4n], f32 circles[3n], u8 cells[w*h]
//   тип 2 «кадр»: заголовок 96 байт (см. encode_frame) + f32 ranges[n_rays]
//                 + u32 n_obstacles + f32 {x, y, radius, vx, vy}[n_obstacles]
// Браузер → сервер: два f32 (v, w) — ручное управление; (0, 0) — отпустить.
#include "roversim/viz.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "viewer_html.hpp"  // kViewerHtml: страница из web/viewer.html, встроенная при сборке

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace roversim::viz {
namespace {

static_assert(std::endian::native == std::endian::little, "the wire format assumes a little-endian host");

constexpr std::size_t kMaxRequestBytes = 1 << 16;      // предел входного буфера соединения
constexpr std::int64_t kTeleopFreshNs = 300'000'000;  // команда с клавиатуры живёт 0.3 с

std::int64_t now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- SHA-1 и Base64: нужны только для рукопожатия WebSocket ----

std::array<std::uint8_t, 20> sha1(const std::string& input) {
    std::uint32_t h[5] = {0x67452301U, 0xEFCDAB89U, 0x98BADCFEU, 0x10325476U, 0xC3D2E1F0U};
    std::string msg = input;
    const std::uint64_t bit_length = static_cast<std::uint64_t>(input.size()) * 8;
    msg.push_back('\x80');
    while (msg.size() % 64 != 56) msg.push_back('\0');
    for (int shift = 56; shift >= 0; shift -= 8) msg.push_back(static_cast<char>((bit_length >> shift) & 0xFF));

    for (std::size_t offset = 0; offset < msg.size(); offset += 64) {
        std::uint32_t w[80];
        for (std::size_t i = 0; i < 16; ++i) {
            const auto* p = reinterpret_cast<const std::uint8_t*>(msg.data() + offset + 4 * i);
            w[i] = (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) | (std::uint32_t{p[2]} << 8) | p[3];
        }
        for (std::size_t i = 16; i < 80; ++i) w[i] = std::rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (std::size_t i = 0; i < 80; ++i) {
            std::uint32_t f = 0, k = 0;
            if (i < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999U;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1U;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDCU;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6U;
            }
            const std::uint32_t t = std::rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = std::rotl(b, 30);
            b = a;
            a = t;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    std::array<std::uint8_t, 20> digest{};
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 4; ++j) digest[4 * i + j] = static_cast<std::uint8_t>(h[i] >> (24 - 8 * j));
    }
    return digest;
}

std::string base64(const std::uint8_t* data, std::size_t size) {
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    for (std::size_t i = 0; i < size; i += 3) {
        const std::uint32_t chunk = (std::uint32_t{data[i]} << 16) |
                                    (i + 1 < size ? std::uint32_t{data[i + 1]} << 8 : 0U) |
                                    (i + 2 < size ? std::uint32_t{data[i + 2]} : 0U);
        out.push_back(kAlphabet[(chunk >> 18) & 63]);
        out.push_back(kAlphabet[(chunk >> 12) & 63]);
        out.push_back(i + 1 < size ? kAlphabet[(chunk >> 6) & 63] : '=');
        out.push_back(i + 2 < size ? kAlphabet[chunk & 63] : '=');
    }
    return out;
}

// ---- Кодирование сообщений ----

class Writer {
 public:
    void u8(std::uint8_t v) { put(v); }
    void u32(std::uint32_t v) { put(v); }
    void f32(double v) { put(static_cast<float>(v)); }
    void bytes(const void* data, std::size_t size) { buffer_.append(static_cast<const char*>(data), size); }
    void zeros(std::size_t count) { buffer_.append(count, '\0'); }
    std::string take() { return std::move(buffer_); }

 private:
    template <typename T>
    void put(T v) {
        char raw[sizeof(T)];
        std::memcpy(raw, &v, sizeof(T));
        buffer_.append(raw, sizeof(T));
    }
    std::string buffer_;
};

/// Оборачивает данные в бинарный кадр WebSocket (сервер → клиент, без маски).
std::string ws_frame(std::string_view payload, std::uint8_t opcode = 0x2) {
    std::string frame;
    frame.reserve(payload.size() + 10);
    frame.push_back(static_cast<char>(0x80 | opcode));  // FIN + код операции
    if (payload.size() < 126) {
        frame.push_back(static_cast<char>(payload.size()));
    } else if (payload.size() < 65536) {
        frame.push_back(126);
        frame.push_back(static_cast<char>(payload.size() >> 8));
        frame.push_back(static_cast<char>(payload.size() & 0xFF));
    } else {
        frame.push_back(127);
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<char>((static_cast<std::uint64_t>(payload.size()) >> shift) & 0xFF));
        }
    }
    frame.append(payload);
    return frame;
}

std::string encode_world(const World& world) {
    Writer w;
    w.u8(1);
    w.zeros(3);
    w.u32(static_cast<std::uint32_t>(world.width()));
    w.u32(static_cast<std::uint32_t>(world.height()));
    w.f32(world.cell_size());
    w.f32(world.origin().x);
    w.f32(world.origin().y);
    w.u32(static_cast<std::uint32_t>(world.boxes().size()));
    w.u32(static_cast<std::uint32_t>(world.circles().size()));
    for (const Aabb& b : world.boxes()) {
        w.f32(b.min.x);
        w.f32(b.min.y);
        w.f32(b.max.x);
        w.f32(b.max.y);
    }
    for (const Circle& c : world.circles()) {
        w.f32(c.center.x);
        w.f32(c.center.y);
        w.f32(c.radius);
    }
    w.bytes(world.cells().data(), world.cells().size());
    return w.take();
}

std::string encode_frame(const Simulator& sim) {
    const RobotState& robot = sim.robot_state();
    const SimConfig& cfg = sim.config();
    const StepResult& result = sim.last_result();
    const double c = std::cos(robot.pose.theta), s = std::sin(robot.pose.theta);
    const Vec2 sensor = robot.pose.position() + rotate({cfg.lidar.mount.x, cfg.lidar.mount.y}, c, s);
    const std::span<const float> ranges = sim.lidar();

    Writer w;
    w.u8(2);  // 0: тип
    w.u8(static_cast<std::uint8_t>((result.collision ? 1 : 0) | (result.success ? 2 : 0) |
                                   (result.terminated ? 4 : 0) | (result.truncated ? 8 : 0)));  // 1: флаги
    w.zeros(2);
    w.u32(static_cast<std::uint32_t>(ranges.size()));  // 4
    w.u32(static_cast<std::uint32_t>(sim.step_count()));  // 8
    w.u32(static_cast<std::uint32_t>(sim.episode()));  // 12
    w.f32(sim.time());  // 16
    w.f32(robot.pose.x);  // 20
    w.f32(robot.pose.y);  // 24
    w.f32(robot.pose.theta);  // 28
    w.f32(sim.goal().x);  // 32
    w.f32(sim.goal().y);  // 36
    w.f32(sim.odom_pose().x);  // 40
    w.f32(sim.odom_pose().y);  // 44
    w.f32(sim.odom_pose().theta);  // 48
    w.f32(robot.velocity.x);  // 52
    w.f32(robot.yaw_rate);  // 56
    w.f32(cfg.robot.radius);  // 60
    w.f32(sensor.x);  // 64: поза лидара в мировой СК
    w.f32(sensor.y);  // 68
    w.f32(robot.pose.theta + cfg.lidar.mount.theta);  // 72
    w.f32(sim.lidar_model().angle_min());  // 76
    w.f32(sim.lidar_model().angle_increment());  // 80
    w.f32(cfg.lidar.range_max);  // 84
    w.f32(result.reward);  // 88
    w.f32(cfg.task.goal_radius);  // 92
    w.bytes(ranges.data(), ranges.size() * sizeof(float));  // 96
    const std::span<const DynamicObstacle> obstacles = sim.obstacles();
    w.u32(static_cast<std::uint32_t>(obstacles.size()));
    for (const DynamicObstacle& obstacle : obstacles) {
        w.f32(obstacle.position.x);
        w.f32(obstacle.position.y);
        w.f32(obstacle.radius);
        w.f32(obstacle.velocity.x);
        w.f32(obstacle.velocity.y);
    }
    return w.take();
}

bool send_all(int fd, std::string_view data) noexcept {
    while (!data.empty()) {
        const ssize_t sent = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
        if (sent <= 0) {
            if (sent < 0 && errno == EINTR) continue;
            return false;  // ошибка или истёк SO_SNDTIMEO: клиент не успевает читать
        }
        data.remove_prefix(static_cast<std::size_t>(sent));
    }
    return true;
}

/// Значение HTTP-заголовка (имя — в нижнем регистре); пусто, если его нет.
std::string header_value(const std::string& request, std::string_view name) {
    std::string lower = request;
    for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    const std::string needle = "\n" + std::string(name) + ":";
    const std::size_t at = lower.find(needle);
    if (at == std::string::npos) return {};
    std::size_t begin = at + needle.size();
    std::size_t end = request.find('\r', begin);
    if (end == std::string::npos) end = request.size();
    while (begin < end && request[begin] == ' ') ++begin;
    while (end > begin && request[end - 1] == ' ') --end;
    return request.substr(begin, end - begin);
}

}  // namespace

struct WebViewer::Impl {
    struct Client {
        int fd = -1;
        bool upgraded = false;  // рукопожатие WebSocket выполнено
        std::string in;         // непрочитанные входящие байты
        std::uint64_t world_gen = 0, frame_gen = 0;
    };

    int listen_fd = -1;
    std::uint16_t port = 0;
    std::int64_t frame_period_ns = 0;
    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<int> viewers{0};
    std::atomic<std::int64_t> next_frame_ns{0};

    // Слот с последними сообщениями: пишет поток симуляции, читает сервер.
    std::mutex mutex;
    std::shared_ptr<const std::string> world_msg, frame_msg;
    std::uint64_t world_gen = 0, frame_gen = 0;
    std::shared_ptr<const World> world;
    std::uint64_t world_revision = 0;

    std::atomic<float> teleop_v{0.0F}, teleop_w{0.0F};
    std::atomic<std::int64_t> teleop_ns{0};

    std::vector<Client> clients;  // только поток сервера

    void run();
    bool on_http(Client& client);
    bool on_frames(Client& client);
    void accept_client();
};

void WebViewer::Impl::accept_client() {
    const int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) return;
    // Тайм-аут записи: зависший клиент отключается, а не блокирует рассылку остальным.
    const timeval timeout{0, 200'000};
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    Client client;
    client.fd = fd;
    clients.push_back(std::move(client));
}

/// Обрабатывает HTTP-запрос. false — соединение нужно закрыть.
bool WebViewer::Impl::on_http(Client& client) {
    const std::size_t end = client.in.find("\r\n\r\n");
    if (end == std::string::npos) return true;  // заголовки пришли не целиком
    const std::string request = client.in.substr(0, end);
    client.in.erase(0, end + 4);

    const std::string key = header_value(request, "sec-websocket-key");
    if (!key.empty()) {
        const auto digest = sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
        const std::string response = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                                     "Sec-WebSocket-Accept: " + base64(digest.data(), digest.size()) + "\r\n\r\n";
        if (!send_all(client.fd, response)) return false;
        client.upgraded = true;
        return on_frames(client);
    }

    const bool root = request.starts_with("GET / ") || request.starts_with("GET /index.html ");
    const std::string_view body = root ? std::string_view(kViewerHtml) : std::string_view("Not found\n");
    const std::string head = std::string("HTTP/1.1 ") + (root ? "200 OK" : "404 Not Found") +
                             "\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: " + std::to_string(body.size()) +
                             "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    if (send_all(client.fd, head)) send_all(client.fd, body);
    ::shutdown(client.fd, SHUT_WR);
    return false;
}

/// Разбирает входящие кадры WebSocket. false — соединение нужно закрыть.
bool WebViewer::Impl::on_frames(Client& client) {
    std::string& in = client.in;
    for (;;) {
        if (in.size() < 2) return true;
        const auto* p = reinterpret_cast<const std::uint8_t*>(in.data());
        const std::uint8_t opcode = p[0] & 0x0F;
        const bool masked = (p[1] & 0x80) != 0;
        std::uint64_t length = p[1] & 0x7F;
        std::size_t header = 2;
        if (length == 126) {
            if (in.size() < 4) return true;
            length = (std::uint64_t{p[2]} << 8) | p[3];
            header = 4;
        } else if (length == 127) {
            return false;  // такие большие сообщения браузеру слать незачем
        }
        if (!masked) return false;  // по RFC 6455 клиент обязан маскировать кадры
        if (in.size() < header + 4 + length) return true;

        std::string payload(in, header + 4, static_cast<std::size_t>(length));
        for (std::size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(payload[i] ^ in[header + (i & 3)]);
        in.erase(0, header + 4 + static_cast<std::size_t>(length));

        if (opcode == 0x8) {  // close
            send_all(client.fd, ws_frame({}, 0x8));
            return false;
        }
        if (opcode == 0x9) {  // ping → pong
            if (!send_all(client.fd, ws_frame(payload, 0xA))) return false;
        } else if (opcode == 0x2 && payload.size() == 2 * sizeof(float)) {
            float command[2];
            std::memcpy(command, payload.data(), sizeof(command));
            if (std::isfinite(command[0]) && std::isfinite(command[1])) {
                const bool released = command[0] == 0.0F && command[1] == 0.0F;
                teleop_v.store(std::clamp(command[0], -1.0F, 1.0F));
                teleop_w.store(std::clamp(command[1], -1.0F, 1.0F));
                teleop_ns.store(released ? 0 : now_ns());
            }
        }
    }
}

void WebViewer::Impl::run() {
    std::vector<pollfd> fds;
    char buffer[4096];
    while (!stop.load(std::memory_order_relaxed)) {
        fds.clear();
        fds.push_back({listen_fd, POLLIN, 0});
        for (const Client& client : clients) fds.push_back({client.fd, POLLIN, 0});
        ::poll(fds.data(), fds.size(), 5);

        // Входящие данные. fds[i + 1] соответствует clients[i].
        const std::size_t polled = clients.size();
        const auto drop = [](Client& client) {
            ::close(client.fd);
            client.fd = -1;
        };
        for (std::size_t i = 0; i < polled; ++i) {
            Client& client = clients[i];
            const short events = fds[i + 1].revents;
            if ((events & POLLIN) != 0) {
                const ssize_t got = ::recv(client.fd, buffer, sizeof(buffer), 0);
                if (got <= 0) {
                    drop(client);
                    continue;
                }
                client.in.append(buffer, static_cast<std::size_t>(got));
                const bool keep = client.in.size() <= kMaxRequestBytes &&
                                  (client.upgraded ? on_frames(client) : on_http(client));
                if (!keep) drop(client);
            } else if ((events & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                drop(client);
            }
        }
        if ((fds[0].revents & POLLIN) != 0) accept_client();

        // Рассылка: каждому зрителю — сначала карта (если она новая), затем свежий кадр.
        std::shared_ptr<const std::string> world_out, frame_out;
        std::uint64_t world_now = 0, frame_now = 0;
        {
            std::lock_guard lock(mutex);
            world_out = world_msg;
            frame_out = frame_msg;
            world_now = world_gen;
            frame_now = frame_gen;
        }
        int count = 0;
        for (Client& client : clients) {
            if (client.fd < 0 || !client.upgraded) continue;
            if (world_out && client.world_gen != world_now) {
                if (!send_all(client.fd, *world_out)) {
                    drop(client);
                    continue;
                }
                client.world_gen = world_now;
            }
            if (frame_out && client.frame_gen != frame_now && client.world_gen == world_now) {
                if (!send_all(client.fd, *frame_out)) {
                    drop(client);
                    continue;
                }
                client.frame_gen = frame_now;
            }
            ++count;
        }
        std::erase_if(clients, [](const Client& client) { return client.fd < 0; });
        viewers.store(count, std::memory_order_relaxed);
    }
    for (Client& client : clients) ::close(client.fd);
    clients.clear();
}

WebViewer::WebViewer(std::uint16_t port, bool bind_all, double max_fps) : impl_(std::make_unique<Impl>()) {
    Impl& s = *impl_;
    s.frame_period_ns = max_fps > 0.0 ? static_cast<std::int64_t>(1e9 / max_fps) : 0;
    s.listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s.listen_fd < 0) throw std::runtime_error(std::string("WebViewer: socket: ") + std::strerror(errno));
    const int one = 1;
    ::setsockopt(s.listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(bind_all ? INADDR_ANY : INADDR_LOOPBACK);
    if (::bind(s.listen_fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(s.listen_fd, 8) != 0) {
        const std::string reason = std::strerror(errno);
        ::close(s.listen_fd);
        throw std::runtime_error("WebViewer: cannot listen on port " + std::to_string(port) + ": " + reason);
    }
    socklen_t length = sizeof(address);
    ::getsockname(s.listen_fd, reinterpret_cast<sockaddr*>(&address), &length);
    s.port = ntohs(address.sin_port);
    s.thread = std::thread([&s] { s.run(); });
}

WebViewer::~WebViewer() {
    impl_->stop.store(true);
    impl_->thread.join();
    ::close(impl_->listen_fd);
}

std::uint16_t WebViewer::port() const noexcept { return impl_->port; }
int WebViewer::num_clients() const noexcept { return impl_->viewers.load(std::memory_order_relaxed); }

void WebViewer::publish(const Simulator& sim, bool force) {
    Impl& s = *impl_;
    // Быстрый путь: зрителей нет либо предыдущий кадр ушёл только что.
    if (s.viewers.load(std::memory_order_relaxed) == 0) return;
    const std::int64_t now = now_ns();
    if (!force && now < s.next_frame_ns.load(std::memory_order_relaxed)) return;
    s.next_frame_ns.store(now + s.frame_period_ns, std::memory_order_relaxed);

    const std::shared_ptr<const World> world = sim.world_ptr();
    bool world_changed = false;
    {
        std::lock_guard lock(s.mutex);
        world_changed = world != s.world || world->revision() != s.world_revision;
    }
    // Сообщения собираются без блокировки; под ней — только обмен указателей.
    std::shared_ptr<const std::string> world_msg;
    if (world_changed) world_msg = std::make_shared<const std::string>(ws_frame(encode_world(*world)));
    auto frame_msg = std::make_shared<const std::string>(ws_frame(encode_frame(sim)));

    std::lock_guard lock(s.mutex);
    if (world_changed) {
        s.world = world;
        s.world_revision = world->revision();
        s.world_msg = std::move(world_msg);
        ++s.world_gen;
    }
    s.frame_msg = std::move(frame_msg);
    ++s.frame_gen;
}

bool WebViewer::teleop(double& v, double& w) const noexcept {
    const Impl& s = *impl_;
    if (now_ns() - s.teleop_ns.load(std::memory_order_relaxed) > kTeleopFreshNs) return false;
    v = s.teleop_v.load(std::memory_order_relaxed);
    w = s.teleop_w.load(std::memory_order_relaxed);
    return true;
}

}  // namespace roversim::viz
