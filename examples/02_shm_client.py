"""Управление симулятором в другом процессе через разделяемую память.

Скрипт запускает ``roversim_server`` (симулятор — отдельный процесс на C++),
подключается к его сегменту разделяемой памяти и ведёт робота к цели полным
навигационным контуром: SLAM + Theta* + DWA. Навигатору хватает лидара и
одометрии, поэтому ему всё равно, где работает симулятор.

Наблюдения читаются прямо из разделяемой памяти: ``client.lidar`` — массив
numpy поверх сегмента, без копирования.

Запуск (из корня проекта, после сборки)::

    PYTHONPATH=build/python python examples/02_shm_client.py
    PYTHONPATH=build/python python examples/02_shm_client.py --viz web   # картинка от сервера
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

import roversim as rs
from roversim import gt, obs


def main() -> None:
    default_server = os.path.join(os.path.dirname(__file__), "..", "build", "roversim_server")
    parser = argparse.ArgumentParser(description="Клиент симулятора через разделяемую память")
    parser.add_argument("--server", default=os.environ.get("ROVERSIM_SERVER", default_server), help="путь к roversim_server")
    parser.add_argument("--shm", default=f"/roversim_example_{os.getpid()}", help="имя сегмента разделяемой памяти")
    parser.add_argument("--world", default="maze", choices=["maze", "arena", "clutter"])
    parser.add_argument("--viz", default="none", choices=["none", "web"], help="визуализация на стороне сервера")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()

    if not os.path.exists(args.server):
        sys.exit(f"не найден {args.server}: соберите проект (cmake --build build) или укажите --server")

    # 1. Симулятор — отдельный процесс.
    # Лимит шагов поднят: путь в неизвестном лабиринте длиннее кратчайшего.
    server = subprocess.Popen([args.server, "--shm", args.shm, "--world", args.world, "--viz", args.viz,
                               "--max-steps", str(args.max_steps), "--no-collision-stop"])
    try:
        # 2. Подключение. Клиент сам дожидается, пока сервер создаст сегмент.
        client = rs.ShmClient(args.shm, timeout=10.0)
        print(f"подключено к {args.shm}: лидар {client.num_rays} лучей, период управления {client.control_dt * 1000:.0f} мс")
        if args.viz == "web":
            print("просмотр: http://127.0.0.1:8765/ (старт через 3 с)")
            time.sleep(3.0)

        client.reset(args.seed)
        lidar, proprio, state = client.lidar, client.proprio, client.state   # виды прямо в разделяемую память
        goal = (float(state[gt.GOAL_X]), float(state[gt.GOAL_Y]))

        # 3. Автономный водитель: только лидар и одометрия.
        agent = rs.NavigationAgent(client)
        agent.set_goal(*goal)

        latencies = []
        result = None
        for step in range(1, args.max_steps + 1):
            left, right = agent.act()
            started = time.perf_counter()
            result = client.step(left, right)          # запрос — ответ через разделяемую память
            latencies.append(time.perf_counter() - started)
            if step % 100 == 0:
                x, y, _ = agent.navigator.pose
                print(f"шаг {step:4d}: оценка позы SLAM ({x:5.2f}, {y:5.2f}), до цели {proprio[obs.GOAL_DISTANCE]:5.2f} м, "
                      f"ближайшее препятствие {float(lidar.min()):.2f} м")
            if args.viz == "web":
                time.sleep(client.control_dt)
            if result.terminated or result.truncated:
                break

        outcome = "цель достигнута" if result.success else ("столкновение" if result.collision else "не успел")
        truth = np.array([state[gt.X], state[gt.Y]])
        slam_error = float(np.linalg.norm(np.array(agent.navigator.pose[:2]) - truth))
        odom_error = float(np.linalg.norm(np.array([proprio[obs.ODOM_X], proprio[obs.ODOM_Y]]) - truth))
        latency_us = 1e6 * np.array(latencies)
        print(f"итог: {outcome} за {step} шагов ({client.time:.1f} с модельного времени)")
        print(f"ошибка позы: SLAM {100 * slam_error:.1f} см, чистая одометрия {100 * odom_error:.1f} см")
        print(f"обмен через разделяемую память: медиана {np.median(latency_us):.0f} мкс на шаг, "
              f"99-й процентиль {np.percentile(latency_us, 99):.0f} мкс (включая сам шаг симуляции)")
        client.close_server()
    finally:
        try:
            server.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server.terminate()


if __name__ == "__main__":
    main()
