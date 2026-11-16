"""Навигация: связка модуля ``roversim.nav`` с симулятором и отрисовка её состояния."""

from __future__ import annotations

import math

import numpy as np

from ._core import RobotParams, nav
from ._core import obs as OBS


def make_navigator(backend, config: "nav.NavigatorConfig | None" = None) -> "nav.Navigator":
    """Навигатор, согласованный с роботом и лидаром данного симулятора.

    ``backend`` — ``Simulator`` или ``ShmClient``. От него берутся только
    паспортные данные (геометрия лидара, размеры и кинематика робота, период
    управления) и начальная поза одометрии, которая задаёт систему координат карты.
    """
    if config is None:
        robot = RobotParams()
        robot.radius = backend.robot_radius
        robot.wheel_radius = backend.wheel_radius
        robot.track_width = backend.track_width
        robot.max_wheel_speed = backend.max_wheel_speed
        config = nav.navigator_config(robot, backend.control_dt)

    angles = backend.lidar_angles
    scan = nav.ScanGeometry()
    scan.num_rays = len(angles)
    scan.angle_min = float(angles[0])
    scan.angle_increment = float(angles[1] - angles[0]) if len(angles) > 1 else 0.0
    scan.range_max = backend.lidar_range_max
    scan.mount = backend.lidar_mount

    p = backend.proprio
    return nav.Navigator(config, scan, (float(p[OBS.ODOM_X]), float(p[OBS.ODOM_Y]), float(p[OBS.ODOM_THETA])))


class NavigationAgent:
    """Автономный водитель: SLAM, глобальный планировщик и DWA по показаниям сенсоров.

    Пример::

        agent = NavigationAgent(sim)
        agent.set_goal(*sim.goal)
        while True:
            result = sim.step(*agent.act())
            if result.terminated or result.truncated:
                break

    Агент не обращается ни к истинной позе, ни к истинной карте: только лидар и
    одометрия. Поэтому он одинаково работает с ``Simulator`` и с ``ShmClient``.
    Создавать его нужно после ``reset()``: начальная поза одометрии становится
    началом отсчёта карты.
    """

    def __init__(self, backend, config: "nav.NavigatorConfig | None" = None) -> None:
        self.backend = backend
        self.navigator = make_navigator(backend, config)
        self._lidar = backend.lidar
        self._proprio = backend.proprio

    def set_goal(self, x: float, y: float) -> None:
        """Цель в системе координат карты (она совпадает с системой одометрии)."""
        self.navigator.set_goal(x, y)

    def act(self) -> tuple[float, float]:
        """Один такт управления: уставки скоростей колёс (левое, правое), рад/с."""
        p = self._proprio
        odom = (float(p[OBS.ODOM_X]), float(p[OBS.ODOM_Y]), float(p[OBS.ODOM_THETA]))
        left, right = self.navigator.update(self._lidar, odom, float(p[OBS.ODOM_V]), float(p[OBS.ODOM_W]))
        return left, right


# ------------------------------------------------------------------- отрисовка


def _disc(image: np.ndarray, x: float, y: float, radius: float, color, outline: bool = False) -> None:
    h, w = image.shape[:2]
    x0, x1 = max(0, int(x - radius - 1)), min(w, int(x + radius + 2))
    y0, y1 = max(0, int(y - radius - 1)), min(h, int(y + radius + 2))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx + 0.5 - x, yy + 0.5 - y)
    mask = (np.abs(d - radius) <= 1.0) if outline else (d <= radius)
    image[y0:y1, x0:x1][mask] = color


def _line(image: np.ndarray, x0: float, y0: float, x1: float, y1: float, color, width: int = 1) -> None:
    h, w = image.shape[:2]
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, steps + 1).astype(int)
    ys = np.linspace(y0, y1, steps + 1).astype(int)
    for dx in range(width):
        for dy in range(width):
            ok = (xs + dx >= 0) & (xs + dx < w) & (ys + dy >= 0) & (ys + dy < h)
            image[ys[ok] + dy, xs[ok] + dx] = color


def render_navigation(navigator: "nav.Navigator", scale: int = 3, truth_pose=None, crop: bool = True) -> np.ndarray:
    """Картина мира глазами навигатора: кадр RGB ``uint8 [H, W, 3]``.

    * карта занятости: белое — свободно, чёрное — занято, серое — не разведано;
    * зелёная линия — глобальный маршрут, зелёная точка — локальная цель;
    * синий круг — оценка позы SLAM, жёлтое кольцо — цель;
    * фиолетовые круги — сопровождаемые подвижные объекты и их скорости;
    * красное кольцо — истинная поза (если передана ``truth_pose``): расхождение
      с синим кругом и есть ошибка локализации.
    """
    grid = navigator.map
    probability = grid.probability()  # [h, w], строка 0 — нижний край
    gray = (255.0 * (1.0 - probability)).astype(np.uint8)
    image = np.repeat(gray[::-1, :, None], 3, axis=2)  # ось Y вверх
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1).copy()

    ox, oy = grid.origin
    px_per_m = scale / grid.resolution
    height = image.shape[0]

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - ox) * px_per_m, height - (y - oy) * px_per_m

    path = navigator.path
    for a, b in zip(path[:-1], path[1:]):
        _line(image, *px(*a), *px(*b), (40, 170, 90), width=2)
    for track in navigator.tracks:
        if not track.moving:
            continue
        cx, cy = px(*track.position)
        _disc(image, cx, cy, track.radius * px_per_m, (186, 104, 200))
        vx, vy = track.velocity
        _line(image, cx, cy, *px(track.position[0] + vx, track.position[1] + vy), (120, 40, 140), width=2)

    gx, gy = px(*navigator.goal)
    _disc(image, gx, gy, 0.3 * px_per_m, (240, 180, 20), outline=True)
    if navigator.status in (1, 4):
        _disc(image, *px(*navigator.carrot), 3, (40, 170, 90))

    x, y, theta = navigator.pose
    radius = navigator.config.dwa.robot_radius * px_per_m
    cx, cy = px(x, y)
    _disc(image, cx, cy, radius, (66, 135, 245))
    _line(image, cx, cy, *px(x + 0.3 * math.cos(theta), y + 0.3 * math.sin(theta)), (255, 255, 255), width=2)
    if truth_pose is not None:
        _disc(image, *px(truth_pose[0], truth_pose[1]), radius, (230, 60, 50), outline=True)

    if crop:
        # Обрезка по разведанной области: карта намного больше освоенной её части.
        known = np.argwhere(np.abs(probability - 0.5) > 0.05)
        if len(known):
            (y0, x0), (y1, x1) = known.min(axis=0), known.max(axis=0)
            pad = int(0.5 / grid.resolution)
            rows = slice(max(0, (probability.shape[0] - 1 - y1 - pad)) * scale, min(probability.shape[0], probability.shape[0] - y0 + pad) * scale)
            cols = slice(max(0, x0 - pad) * scale, min(probability.shape[1], x1 + 1 + pad) * scale)
            image = image[rows, cols]
    return np.ascontiguousarray(image)
