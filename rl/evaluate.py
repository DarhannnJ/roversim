"""Проверка обученной политики на лабиринтах, которых не было в обучении.

Примеры (из корня проекта)::

    # метрики по 500 эпизодам
    PYTHONPATH=build/python python rl/evaluate.py models/ppo_rover.pt --episodes 500

    # посмотреть езду в браузере (http://127.0.0.1:8765/)
    PYTHONPATH=build/python python rl/evaluate.py models/ppo_rover.pt --web

    # сохранить картинку с траекторией одного эпизода
    PYTHONPATH=build/python python rl/evaluate.py models/ppo_rover.pt --png episode.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import roversim as rs
from model import encode_observation, evaluate_policy, load_checkpoint, make_mazes, make_sim_config, to_env_action


def run_single(model, checkpoint, args) -> None:
    """Один эпизод в одиночной среде Gymnasium — с отрисовкой."""
    sim = dict(checkpoint["sim"])
    if args.obstacles is not None:
        sim["obstacles.count"] = args.obstacles
    world = make_mazes(1, first_seed=2_000_000 + args.seed, world=checkpoint["world"])[0]
    render_mode = "web" if args.web else "rgb_array"
    env = rs.RoverEnv(world, make_sim_config(sim), action_mode="twist", render_mode=render_mode, render_size=640)
    if args.web:
        print("просмотр: http://127.0.0.1:8765/ (ожидание 3 с, чтобы успеть открыть страницу)")
        time.sleep(3.0)

    obs, _ = env.reset(seed=args.seed)
    done, steps, total = False, 0, 0.0
    while not done:
        scan, features = encode_observation(obs["lidar"][None], obs["proprio"][None], env.max_linear_speed, env.max_angular_speed)
        with torch.no_grad():
            action, _, _ = model.act(scan, features, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(to_env_action(action)[0])
        done = terminated or truncated
        steps += 1
        total += reward
        if args.web:
            time.sleep(env.sim.control_dt)  # реальное время
    outcome = "цель достигнута" if info["is_success"] else ("столкновение" if info["collision"] else "не успел")
    print(f"эпизод: {outcome}, шагов {steps} ({steps * env.sim.control_dt:.1f} с), сумма наград {total:.2f}")
    if args.png:
        from PIL import Image

        Image.fromarray(env.render()).save(args.png)
        print(f"кадр сохранён: {args.png}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка обученной политики навигации")
    parser.add_argument("checkpoint", help="файл модели, сохранённый train_ppo.py")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--mazes", type=int, default=64, help="число новых лабиринтов")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obstacles", type=int, default=None, help="подвижных препятствий (по умолчанию — как при обучении)")
    parser.add_argument("--stochastic", action="store_true", help="выбирать действия случайно, а не брать среднее политики")
    parser.add_argument("--web", action="store_true", help="показать один эпизод в браузере")
    parser.add_argument("--png", default=None, help="сохранить кадр конца одного эпизода в файл")
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    print(f"модель: {args.checkpoint}, лидар {checkpoint['num_rays']} лучей, лабиринты {checkpoint['world']}")
    if args.web or args.png:
        run_single(model, checkpoint, args)
        return

    sim = dict(checkpoint["sim"])
    if args.obstacles is not None:
        sim["obstacles.count"] = args.obstacles
    # Лабиринты для проверки: seed генератора далеко от обучающих.
    worlds = make_mazes(args.mazes, first_seed=3_000_000 + args.seed, world=checkpoint["world"])
    started = time.time()
    metrics = evaluate_policy(model, worlds, sim, args.episodes, seed=args.seed + 99, deterministic=not args.stochastic)
    print(f"эпизодов: {metrics['episodes']} на {args.mazes} новых лабиринтах ({time.time() - started:.1f} с)")
    print(f"  цель достигнута : {100 * metrics['success_rate']:.1f}%")
    print(f"  столкновение    : {100 * metrics['collision_rate']:.1f}%")
    print(f"  не успел        : {100 * metrics['timeout_rate']:.1f}%")
    print(f"  SPL             : {metrics['spl']:.3f}")
    print(f"  средняя длина   : {metrics['mean_steps']:.0f} шагов")


if __name__ == "__main__":
    main()
