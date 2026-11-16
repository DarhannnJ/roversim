"""Автономная навигация в неизвестной среде: SLAM + Theta* + DWA.

Робот не знает ни карты, ни своей истинной позы. По лидару и шумной одометрии он

* строит карту занятости и уточняет по ней свою позу (сеточный SLAM);
* прокладывает глобальный маршрут по уже построенной части карты (Theta*),
  считая неразведанное проходимым, и перестраивает его, когда открываются стены;
* объезжает препятствия локальным планировщиком DWA, причём подвижные —
  с прогнозом их движения (скорости оценивает трекер по сканам).

В конце сохраняется картинка: слева истинный мир, справа карта, построенная роботом.

Запуск (из корня проекта, после сборки)::

    PYTHONPATH=build/python python examples/04_slam_navigation.py
    PYTHONPATH=build/python python examples/04_slam_navigation.py --obstacles 4 --episodes 5
    PYTHONPATH=build/python python examples/04_slam_navigation.py --web     # http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

import roversim as rs
from roversim import gt, obs

STATUS = {0: "нет цели", 1: "едет", 2: "цель достигнута", 3: "нет маршрута", 4: "восстановление"}


def run_episode(sim: rs.Simulator, seed: int, viewer: "rs.WebViewer | None", verbose: bool):
    sim.reset(seed)
    agent = rs.NavigationAgent(sim)     # создаётся после reset(): начало карты — стартовая поза одометрии
    agent.set_goal(*sim.goal)           # цель в системе координат карты
    navigator = agent.navigator
    state = sim.state

    contacts, travelled, worst_slam = 0, 0.0, 0.0
    previous = np.array(sim.pose[:2])
    started = time.perf_counter()
    while True:
        result = sim.step(*agent.act())
        position = np.array([state[gt.X], state[gt.Y]])
        travelled += float(np.linalg.norm(position - previous))
        previous = position
        contacts += result.collision
        worst_slam = max(worst_slam, float(np.linalg.norm(np.array(navigator.pose[:2]) - position)))
        if viewer is not None:
            viewer.publish(sim)
            time.sleep(sim.control_dt)
        if verbose and sim.step_count % 100 == 0:
            moving = sum(track.moving for track in navigator.tracks)
            print(f"  t={sim.time:5.1f} с  {STATUS[navigator.status]:<15} до цели {sim.proprio[obs.GOAL_DISTANCE]:5.2f} м  "
                  f"маршрут {len(navigator.path)} точек  перепланирований {navigator.replans}  "
                  f"подвижных объектов в сопровождении {moving}")
        if result.terminated or result.truncated:
            break
    elapsed = time.perf_counter() - started

    shortest = sim.shortest_path_length
    odom = np.array([sim.proprio[obs.ODOM_X], sim.proprio[obs.ODOM_Y]])
    return {
        "success": result.success,
        "steps": sim.step_count,
        "contacts": contacts,
        "spl": shortest / max(shortest, travelled) if result.success else 0.0,
        "slam_error": float(np.linalg.norm(np.array(navigator.pose[:2]) - previous)),
        "worst_slam_error": worst_slam,
        "odom_error": float(np.linalg.norm(odom - previous)),
        "rate": sim.step_count / elapsed,
        "agent": agent,
    }


def save_picture(sim: rs.Simulator, agent: rs.NavigationAgent, path: str) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow не установлен — картинка не сохранена")
        return
    truth = rs.Renderer(640, 640).render(sim)
    belief = rs.render_navigation(agent.navigator, scale=4, truth_pose=sim.pose)
    belief = np.asarray(Image.fromarray(belief).resize((int(belief.shape[1] * 640 / belief.shape[0]), 640), Image.NEAREST))
    canvas = np.full((640, truth.shape[1] + belief.shape[1] + 12, 3), 30, dtype=np.uint8)
    canvas[:, :truth.shape[1]] = truth
    canvas[:, truth.shape[1] + 12:] = belief
    Image.fromarray(canvas).save(path)
    print(f"картинка сохранена: {path} (слева истинный мир, справа карта робота)")


def main() -> None:
    parser = argparse.ArgumentParser(description="SLAM и навигация в неизвестной среде")
    parser.add_argument("--world", default="maze", choices=["maze", "arena", "clutter"])
    parser.add_argument("--obstacles", type=int, default=3, help="подвижных препятствий")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--web", action="store_true", help="живой просмотр истинного мира в браузере")
    parser.add_argument("--png", default="slam_navigation.png")
    args = parser.parse_args()

    config = rs.make_config({
        "task.max_steps": 4000,
        "task.terminate_on_collision": False,   # касание не обрывает эпизод, а только считается
        "obstacles.count": args.obstacles,
    })
    viewer = rs.WebViewer(8765) if args.web else None
    if viewer is not None:
        print("просмотр: http://127.0.0.1:8765/ (старт через 3 с)")
        time.sleep(3.0)

    results = []
    for episode in range(args.episodes):
        world = rs.make_world(args.world, seed=100 + args.seed + episode)   # новый мир на каждый эпизод
        sim = rs.Simulator(config, world)
        print(f"эпизод {episode + 1}: мир '{args.world}' №{100 + args.seed + episode}, подвижных препятствий {args.obstacles}")
        result = run_episode(sim, args.seed + episode, viewer, verbose=True)
        results.append(result)
        print(f"  итог: {'цель достигнута' if result['success'] else 'цель НЕ достигнута'} за {result['steps']} шагов, "
              f"касаний {result['contacts']}, SPL {result['spl']:.2f}; "
              f"ошибка позы SLAM {100 * result['slam_error']:.1f} см (макс. {100 * result['worst_slam_error']:.1f} см), "
              f"одометрии {100 * result['odom_error']:.1f} см; {result['rate']:.0f} шагов/с")
        if episode == args.episodes - 1 and args.png:
            save_picture(sim, result["agent"], args.png)

    print(f"всего: целей достигнуто {sum(r['success'] for r in results)}/{len(results)}, "
          f"эпизодов без касаний {sum(r['contacts'] == 0 for r in results)}/{len(results)}, "
          f"средний SPL {np.mean([r['spl'] for r in results]):.2f}, "
          f"средняя ошибка SLAM {100 * np.mean([r['slam_error'] for r in results]):.1f} см "
          f"против {100 * np.mean([r['odom_error'] for r in results]):.1f} см у одометрии")


if __name__ == "__main__":
    main()
