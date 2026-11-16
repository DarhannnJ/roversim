"""Быстрый старт: подключиться к ядру, управлять роботом и читать лидар.

Робот едет по лабиринту к цели под управлением простого реактивного регулятора:

* направление движения задаёт подсказка маршрута из наблюдения
  (``PATH_COS`` / ``PATH_SIN`` — куда ведёт кратчайший путь);
* скорость ограничивается свободной дальностью лидара впереди;
* стена сбоку ближе полуметра добавляет небольшой отворот от неё.

Запуск (из корня проекта, после сборки)::

    PYTHONPATH=build/python python examples/01_quickstart.py
    PYTHONPATH=build/python python examples/01_quickstart.py --web       # смотреть в браузере
    PYTHONPATH=build/python python examples/01_quickstart.py --obstacles 3
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import roversim as rs
from roversim import gt, obs


def controller(lidar: np.ndarray, angles: np.ndarray, proprio: np.ndarray, max_yaw_rate: float) -> np.ndarray:
    """Действие среды: (линейная, угловая) скорости в долях максимума."""
    bearing = math.atan2(proprio[obs.PATH_SIN], proprio[obs.PATH_COS])

    # Свободная дальность в секторе ±20° по курсу и ближайшие стены слева и справа.
    front = float(lidar[np.abs(angles) < 0.35].min())
    left = float(lidar[(angles > 0.5) & (angles < 1.6)].min())
    right = float(lidar[(angles < -0.5) & (angles > -1.6)].min())

    # Вперёд — тем быстрее, чем дальше препятствие и чем точнее курс; при большой
    # ошибке курса робот сначала разворачивается на месте.
    speed = np.clip((front - 0.30) / 0.8, 0.0, 0.8) * max(0.0, math.cos(bearing))
    if abs(bearing) > 0.7:
        speed = 0.0
    # Поворот: к направлению маршрута плюс отталкивание от близкой боковой стены.
    repulsion = 0.6 * (max(0.0, 0.5 - right) - max(0.0, 0.5 - left))
    yaw_rate = np.clip(2.0 * bearing + repulsion, -2.5, 2.5)
    return np.array([speed, yaw_rate / max_yaw_rate], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Быстрый старт roversim")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--world", default="maze", choices=["maze", "arena", "clutter"])
    parser.add_argument("--obstacles", type=int, default=0, help="подвижных препятствий")
    parser.add_argument("--web", action="store_true", help="живой просмотр: http://127.0.0.1:8765/")
    parser.add_argument("--png", default="quickstart.png", help="куда сохранить итоговый кадр")
    args = parser.parse_args()

    # 1. Мир и конфигурация. Любое поле задаётся путём через точку.
    world = rs.make_world(args.world, seed=1)
    config = rs.make_config({
        "lidar.num_rays": 360,
        "task.max_steps": 1200,
        "task.spawn_clearance": 0.18,   # маршрут-подсказка держится подальше от стен
        "obstacles.count": args.obstacles,
    })

    # 2. Среда Gymnasium. Действие — скорости в долях максимума, наблюдение —
    #    словарь из скана лидара и вектора proprio (IMU, одометрия, цель).
    env = rs.RoverEnv(world, config, action_mode="twist", obs_mode="dict",
                      render_mode="web" if args.web else "rgb_array")
    angles = env.lidar_angles   # углы лучей в системе координат робота
    print(f"мир: {world.width}×{world.height} клеток по {world.cell_size} м; "
          f"лидар: {len(angles)} лучей, {math.degrees(angles[1] - angles[0]):.1f}° на луч; "
          f"шаг управления {env.sim.control_dt * 1000:.0f} мс")
    if args.web:
        print("просмотр: http://127.0.0.1:8765/ (старт через 3 с)")
        time.sleep(3.0)

    observation, info = env.reset(seed=args.seed)
    print(f"кратчайший путь до цели: {env.sim.shortest_path_length:.2f} м")

    done, step, total_reward = False, 0, 0.0
    while not done:
        # 3. Данные лидара — обычный массив numpy: дальности в метрах.
        lidar, proprio = observation["lidar"], observation["proprio"]
        action = controller(lidar, angles, proprio, env.max_angular_speed)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        if step % 40 == 0 or done:
            nearest = int(np.argmin(lidar))
            state = env.state   # истинное состояние (для отладки), индексы — roversim.gt
            print(f"t={info['time']:5.1f} с  поза=({state[gt.X]:5.2f}, {state[gt.Y]:5.2f}, {math.degrees(state[gt.THETA]):6.1f}°)  "
                  f"до цели {proprio[obs.GOAL_DISTANCE]:5.2f} м  "
                  f"ближайшее препятствие {lidar[nearest]:.2f} м под {math.degrees(angles[nearest]):6.1f}°  "
                  f"скорость {proprio[obs.ODOM_V]:.2f} м/с")
        if args.web:
            time.sleep(env.sim.control_dt)

    outcome = "цель достигнута" if info["is_success"] else ("столкновение" if info["collision"] else "лимит шагов")
    print(f"итог: {outcome} за {step} шагов ({info['time']:.1f} с), сумма наград {total_reward:.2f}")

    if not args.web and args.png:
        try:
            from PIL import Image
        except ImportError:
            print("Pillow не установлен — кадр не сохранён")
        else:
            Image.fromarray(env.render()).save(args.png)
            print(f"кадр сохранён: {args.png}")
    env.close()


if __name__ == "__main__":
    main()
