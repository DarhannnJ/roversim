// Отладочная визуализация. Симулятор от неё не зависит: без вызовов из этого
// файла он работает в headless-режиме.
//
//   * Renderer  — программный растеризатор вида сверху (карта, лучи лидара,
//                 робот, цель, след). Кадр годится как rgb_array для Gymnasium
//                 и для записи видео.
//   * to_ansi   — тот же кадр для терминала (TUI): работает по SSH.
//   * WebViewer — HTTP/WebSocket-сервер без внешних зависимостей: живая картинка
//                 в браузере и ручное управление с клавиатуры.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "roversim/simulator.hpp"

namespace roversim::viz {

/// RGB-изображение, 8 бит на канал, строки сверху вниз.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;  ///< width * height * 3 байт
};

class Renderer {
 public:
    Renderer(int width, int height);

    /// Рисует текущее состояние симулятора. Ссылка действительна до следующего вызова.
    const Image& render(const Simulator& sim);

 private:
    void rebuild_background(const World& world);

    Image frame_;
    Image background_;  ///< статичный слой: карта, перерисовывается только при смене мира
    const World* world_ = nullptr;
    std::uint64_t world_revision_ = 0;
    double scale_ = 1.0;  ///< пикселей на метр
    double offset_x_ = 0.0, offset_y_ = 0.0;
    std::vector<Vec2> trail_;  ///< пройденный в этом эпизоде путь
    std::uint64_t episode_ = 0;
};

/// Кадр для терминала: символы «▀» с 24-битным цветом, два пикселя на знакоместо.
/// Строка не содержит команд управления курсором.
std::string to_ansi(const Image& image);

/// Сохраняет кадр в формате PPM (P6). false — файл не удалось записать.
bool write_ppm(const Image& image, const std::string& path);

class WebViewer {
 public:
    /// Запускает сервер в фоновом потоке. port = 0 — любой свободный порт.
    /// По умолчанию слушает только 127.0.0.1; bind_all открывает доступ из сети.
    explicit WebViewer(std::uint16_t port = 8765, bool bind_all = false, double max_fps = 30.0);
    ~WebViewer();

    WebViewer(const WebViewer&) = delete;
    WebViewer& operator=(const WebViewer&) = delete;

    std::uint16_t port() const noexcept;
    int num_clients() const noexcept;

    /// Публикует состояние симулятора. Вызывать можно после каждого шага: если
    /// зрителей нет или предыдущий кадр отправлен недавно, вызов стоит десятки
    /// наносекунд. force — отправить кадр в любом случае (например, последний).
    void publish(const Simulator& sim, bool force = false);

    /// Ручное управление из браузера: v и w в долях максимальных скоростей.
    /// false, если клавиши не нажаты (свежей команды нет).
    bool teleop(double& v, double& w) const noexcept;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace roversim::viz
