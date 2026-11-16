"""Обучение политики навигации методом PPO (Proximal Policy Optimization).

Агент учится проезжать случайные лабиринты до цели без столкновений. Он видит
скан лидара, показания одометрии, направление на цель и направление, в котором
к цели ведёт кратчайший путь (подсказка глобального планировщика); управляет
линейной и угловой скоростью.

Запуск (из корня проекта, после сборки)::

    PYTHONPATH=build/python python rl/train_ppo.py --total-steps 3000000 --save models/ppo_rover.pt

Обучение идёт на пакете из сотен сред, которые шагают параллельно в потоках
ядра; наблюдения попадают в PyTorch без копирования. Каждые несколько
обновлений политика проверяется на лабиринтах, которых не было в обучении.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import roversim as rs
from model import (ActorCritic, encode_observation, evaluate_policy, make_mazes, make_sim_config, save_checkpoint,
                   to_env_action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--total-steps", type=int, default=3_000_000, help="сколько шагов среды сделать всего")
    parser.add_argument("--num-envs", type=int, default=256, help="число параллельных сред")
    parser.add_argument("--rollout", type=int, default=64, help="шагов на среду между обновлениями политики")
    parser.add_argument("--epochs", type=int, default=4, help="проходов по собранным данным за обновление")
    parser.add_argument("--minibatches", type=int, default=8, help="мини-пакетов на проход")
    parser.add_argument("--lr", type=float, default=3e-4, help="начальная скорость обучения (линейно убывает до нуля)")
    parser.add_argument("--gamma", type=float, default=0.99, help="коэффициент дисконтирования")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="параметр обобщённой оценки преимущества (GAE)")
    parser.add_argument("--clip", type=float, default=0.2, help="порог отсечения отношения вероятностей PPO")
    parser.add_argument("--entropy", type=float, default=0.003, help="вес энтропийной добавки")
    parser.add_argument("--value-coef", type=float, default=0.5, help="вес ошибки функции ценности")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="ограничение нормы градиента")
    parser.add_argument("--reward-scale", type=float, default=0.2, help="множитель наград (для масштаба функции ценности)")
    parser.add_argument("--hidden", type=int, default=128, help="ширина полносвязных слоёв")
    parser.add_argument("--seed", type=int, default=1)
    # Мир
    parser.add_argument("--train-mazes", type=int, default=64, help="число разных лабиринтов для обучения")
    parser.add_argument("--maze-cols", type=int, default=5)
    parser.add_argument("--maze-rows", type=int, default=5)
    parser.add_argument("--corridor", type=float, default=1.2, help="ширина проходов [м]")
    parser.add_argument("--loop-prob", type=float, default=0.15, help="доля пробитых стен (петли в лабиринте)")
    parser.add_argument("--obstacles", type=int, default=0, help="подвижных препятствий в каждом эпизоде")
    # Оценка и сохранение
    parser.add_argument("--eval-every", type=int, default=20, help="раз в сколько обновлений проверять на новых лабиринтах")
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--eval-mazes", type=int, default=32)
    parser.add_argument("--save", default="models/ppo_rover.pt", help="куда сохранять веса лучшей модели")
    parser.add_argument("--threads", type=int, default=0, help="потоков симулятора (0 — по числу ядер)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)

    world = {"cols": args.maze_cols, "rows": args.maze_rows, "corridor": args.corridor, "loop_prob": args.loop_prob}
    sim_overrides = {"obstacles.count": args.obstacles}
    config = make_sim_config(sim_overrides)

    # Обучающие и проверочные лабиринты не пересекаются: у них разные seed генератора.
    train_worlds = make_mazes(args.train_mazes, first_seed=0, world=world)
    eval_worlds = make_mazes(args.eval_mazes, first_seed=1_000_000, world=world)
    env = rs.RoverVecEnv(args.num_envs, train_worlds, config, action_mode="twist", num_threads=args.threads)
    num_envs, horizon = args.num_envs, args.rollout
    batch = num_envs * horizon
    updates = max(1, args.total_steps // batch)

    model = ActorCritic(env.num_rays, args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    print(f"среды: {num_envs} × {horizon} шагов = {batch} переходов на обновление, обновлений: {updates}")
    print(f"лидар: {env.num_rays} лучей, параметров сети: {sum(p.numel() for p in model.parameters())}, "
          f"потоков симулятора: {env.vec.num_threads}")

    # Буферы траекторий: [шаг, среда, ...].
    scans = torch.zeros(horizon, num_envs, env.num_rays)
    features = torch.zeros(horizon, num_envs, 7)
    actions = torch.zeros(horizon, num_envs, 2)
    log_probs = torch.zeros(horizon, num_envs)
    rewards = torch.zeros(horizon, num_envs)
    dones = torch.zeros(horizon, num_envs)
    values = torch.zeros(horizon, num_envs)

    obs, _ = env.reset(args.seed)
    episode_return = np.zeros(num_envs)
    recent = {"return": [], "success": [], "collision": [], "length": []}
    episode_length = np.zeros(num_envs, dtype=np.int64)
    best_success, started, env_steps = -1.0, time.time(), 0
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)

    for update in range(1, updates + 1):
        # Скорость обучения линейно убывает: к концу обучения шаги становятся осторожнее.
        lr = args.lr * (1.0 - (update - 1) / updates)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---- 1. Сбор траекторий текущей политикой ----
        for t in range(horizon):
            scan, feat = encode_observation(obs["lidar"], obs["proprio"], env.max_linear_speed, env.max_angular_speed)
            with torch.no_grad():
                action, log_prob, value = model.act(scan, feat)
            scans[t], features[t], actions[t], log_probs[t], values[t] = scan, feat, action, log_prob, value

            obs, reward, terminated, truncated, info = env.step(to_env_action(action))
            done = terminated | truncated
            rewards[t] = torch.from_numpy(reward) * args.reward_scale
            dones[t] = torch.from_numpy(done.astype(np.float32))

            episode_return += reward
            episode_length += 1
            for i in np.nonzero(done)[0]:
                recent["return"].append(episode_return[i])
                recent["length"].append(episode_length[i])
                recent["success"].append(float(info["is_success"][i]))
                recent["collision"].append(float(info["collision"][i]))
                episode_return[i] = 0.0
                episode_length[i] = 0
        env_steps += batch

        # ---- 2. Обобщённая оценка преимущества (GAE) ----
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t);  A_t = delta_t + gamma * lambda * A_{t+1}.
        # На границе эпизода цепочка обрывается: следующее состояние уже принадлежит новому эпизоду.
        with torch.no_grad():
            scan, feat = encode_observation(obs["lidar"], obs["proprio"], env.max_linear_speed, env.max_angular_speed)
            next_value = model.value(scan, feat)
        advantages = torch.zeros(horizon, num_envs)
        last = torch.zeros(num_envs)
        for t in reversed(range(horizon)):
            not_done = 1.0 - dones[t]
            following = next_value if t == horizon - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * following * not_done - values[t]
            last = delta + args.gamma * args.gae_lambda * not_done * last
            advantages[t] = last
        returns = advantages + values

        # ---- 3. Обновление политики и критика ----
        flat = lambda x: x.reshape(batch, *x.shape[2:])  # noqa: E731
        b_scans, b_features, b_actions = flat(scans), flat(features), flat(actions)
        b_log_probs, b_advantages, b_returns = flat(log_probs), flat(advantages), flat(returns)
        minibatch = batch // args.minibatches
        stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0}
        for _ in range(args.epochs):
            order = torch.randperm(batch)
            for start in range(0, batch, minibatch):
                idx = order[start:start + minibatch]
                new_log_prob, entropy, value = model.evaluate(b_scans[idx], b_features[idx], b_actions[idx])
                advantage = b_advantages[idx]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                # Отсечённая суррогатная цель PPO: отношение вероятностей новой и старой
                # политики не может увести оценку дальше чем на clip от единицы.
                log_ratio = new_log_prob - b_log_probs[idx]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(ratio * advantage, ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * advantage).mean()
                value_loss = 0.5 * (value - b_returns[idx]).pow(2).mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                stats["policy"] += policy_loss.item()
                stats["value"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["kl"] += ((ratio - 1.0) - log_ratio).mean().item()
        count = args.epochs * args.minibatches

        # ---- 4. Журнал, проверка, сохранение ----
        if update % 5 == 0 or update == 1 or update == updates:
            window = {key: (float(np.mean(v[-2000:])) if v else float("nan")) for key, v in recent.items()}
            print(f"[{update:4d}/{updates}] шагов {env_steps / 1e6:5.2f} млн | {env_steps / (time.time() - started):6.0f} шаг/с | "
                  f"награда {window['return']:7.2f} | успех {100 * window['success']:5.1f}% | "
                  f"столкн. {100 * window['collision']:5.1f}% | длина {window['length']:5.0f} | "
                  f"энтропия {stats['entropy'] / count:5.2f} | KL {stats['kl'] / count:.4f}", flush=True)
            for v in recent.values():
                del v[:-4000]

        if update % args.eval_every == 0 or update == updates:
            model.eval()
            metrics = evaluate_policy(model, eval_worlds, sim_overrides, args.eval_episodes, seed=args.seed + 777)
            model.train()
            marker = ""
            if metrics["success_rate"] > best_success:
                best_success = metrics["success_rate"]
                save_checkpoint(args.save, model, sim_overrides, world,
                                extra={"env_steps": env_steps, "metrics": metrics, "args": vars(args)})
                marker = f"  -> сохранено в {args.save}"
            print(f"    проверка на {args.eval_mazes} новых лабиринтах ({metrics['episodes']} эпизодов): "
                  f"успех {100 * metrics['success_rate']:.1f}%, столкновения {100 * metrics['collision_rate']:.1f}%, "
                  f"не успел {100 * metrics['timeout_rate']:.1f}%, SPL {metrics['spl']:.3f}{marker}", flush=True)

    print(f"готово: {env_steps} шагов за {time.time() - started:.0f} с, лучшая доля успеха на новых лабиринтах "
          f"{100 * best_success:.1f}%, модель: {args.save}")


if __name__ == "__main__":
    main()
