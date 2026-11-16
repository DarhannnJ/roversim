"""Пакетная симуляция для обучения с подкреплением: скорость и работа без копирования.

Сотни сред шагают параллельно в потоках ядра. Наблюдения всех сред лежат в
общих массивах ``[N, ...]``, которые напрямую становятся тензорами PyTorch:
``torch.from_numpy`` не копирует данные, тензор смотрит в память симулятора.

Запуск (из корня проекта, после сборки)::

    PYTHONPATH=build/python python examples/03_vec_rollout.py
    PYTHONPATH=build/python python examples/03_vec_rollout.py --envs 512 --rays 90
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

import roversim as rs

# Буферы симулятора отдаются «только для чтения», и PyTorch предупреждает, что
# писать в тензор поверх них нельзя. Здесь в них никто и не пишет.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Пакетная симуляция roversim")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--rays", type=int, default=360)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--mazes", type=int, default=16, help="число разных лабиринтов")
    args = parser.parse_args()

    env = rs.RoverVecEnv(args.envs, world="maze", num_worlds=args.mazes, config={"lidar.num_rays": args.rays})
    obs, _ = env.reset(seed=0)
    print(f"{args.envs} сред в {args.mazes} лабиринтах, {env.vec.num_threads} потоков; "
          f"наблюдения: lidar {obs['lidar'].shape} {obs['lidar'].dtype}, proprio {obs['proprio'].shape}")

    # Наблюдения — виды на буферы ядра: адрес памяти не меняется от шага к шагу.
    address = obs["lidar"].ctypes.data
    rng = np.random.default_rng(0)

    try:
        import torch
    except ImportError:
        torch = None
        print("PyTorch не установлен: действия выбираются случайно через numpy")
    else:
        # Тензор поверх той же памяти: изменения после step() видны в нём сразу.
        lidar_tensor = torch.from_numpy(obs["lidar"])
        print(f"тензор PyTorch делит память с симулятором: {lidar_tensor.data_ptr() == address}")

    episodes = successes = collisions = 0
    started = time.perf_counter()
    for _ in range(args.steps):
        actions = rng.uniform(-1.0, 1.0, size=(args.envs, 2))
        actions[:, 0] = np.abs(actions[:, 0])   # случайная езда вперёд
        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated | truncated
        episodes += int(done.sum())
        successes += int((done & info["is_success"]).sum())
        collisions += int((done & info["collision"]).sum())
    elapsed = time.perf_counter() - started
    assert obs["lidar"].ctypes.data == address   # те же буферы: ни одного копирования наблюдений

    total = args.envs * args.steps
    print(f"{total} шагов за {elapsed:.2f} с: {total / elapsed:,.0f} шагов/с "
          f"({total * env.config.control_dt / elapsed:,.0f}× быстрее реального времени)")
    print(f"эпизодов завершено: {episodes} (случайная политика: столкновений {collisions}, целей {successes})")


if __name__ == "__main__":
    main()
