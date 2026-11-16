"""Политика и критик для навигации по лидару, общие для обучения и оценки.

Здесь же зафиксировано всё, что должно совпадать при обучении и при запуске
обученной модели: конфигурация симулятора, преобразование наблюдений в
признаки и преобразование выхода сети в команду роботу.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

import roversim as rs
from roversim import obs as OBS

# Буферы наблюдений симулятора отдаются как массивы «только для чтения», и PyTorch
# предупреждает, что запись в такой тензор недопустима. Запись здесь не выполняется:
# признаки всегда строятся новыми тензорами, так что предупреждение не по делу.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# Дальности лидара обрезаются: для объезда препятствий важна ближняя зона, а
# различие между «стена в 5 м» и «стена в 9 м» политике не нужно.
LIDAR_CLIP = 4.0
# Предельная угловая скорость, которой управляет политика [доля от максимума].
# Полный диапазон (6.7 рад/с) делает управление слишком резким.
YAW_SCALE = 0.45
PROPRIO_FEATURES = 7

# Настройки симулятора для обучения. Те же значения сохраняются в файле модели
# и используются при оценке.
DEFAULT_SIM: dict[str, Any] = {
    "lidar.num_rays": 90,            # 4° на луч: достаточно для коридоров, втрое быстрее 360 лучей
    "task.max_steps": 600,           # 30 секунд на эпизод
    "task.terminate_on_collision": True,
    "task.spawn_clearance": 0.15,    # маршрут-подсказка идёт не ближе 15 см к стенам (плюс радиус)
    "task.lookahead": 0.8,
}


def make_sim_config(overrides: Mapping[str, Any] | None = None) -> rs.SimConfig:
    settings = dict(DEFAULT_SIM)
    settings.update(overrides or {})
    return rs.make_config(settings)


def encode_observation(lidar: np.ndarray, proprio: np.ndarray, max_linear: float, max_angular: float):
    """Наблюдение симулятора → тензоры признаков.

    ``lidar`` и ``proprio`` — массивы симулятора ``[N, ...]`` (обычно виды без
    копирования). ``torch.from_numpy`` тоже не копирует, поэтому первая и
    единственная копия данных возникает при нормировке.
    """
    scan = torch.from_numpy(lidar).clamp(max=LIDAR_CLIP) / LIDAR_CLIP
    p = torch.from_numpy(proprio)
    features = torch.stack(
        (
            p[:, OBS.ODOM_V] / max_linear,
            p[:, OBS.ODOM_W] / max_angular,
            p[:, OBS.GOAL_DISTANCE].clamp(max=10.0) / 10.0,
            p[:, OBS.GOAL_COS],
            p[:, OBS.GOAL_SIN],
            p[:, OBS.PATH_COS],   # направление, в котором кратчайший путь ведёт к цели
            p[:, OBS.PATH_SIN],
        ),
        dim=1,
    )
    return scan, features


def to_env_action(action: torch.Tensor) -> np.ndarray:
    """Выход политики → действие среды (режим ``twist``: доли предельных скоростей).

    Первая компонента отображается из [-1, 1] в [0, 1]: робот едет только вперёд,
    задний ход политике недоступен. Вторая — поворот, ограниченный ``YAW_SCALE``.
    """
    a = action.detach().clamp(-1.0, 1.0)
    out = torch.stack(((a[:, 0] + 1.0) * 0.5, a[:, 1] * YAW_SCALE), dim=1)
    return out.numpy()


def _layer(module: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    nn.init.orthogonal_(module.weight, gain)
    nn.init.zeros_(module.bias)
    return module


class ActorCritic(nn.Module):
    """Гауссова политика и функция ценности с общим кодировщиком.

    Скан лидара — кольцевой сигнал (последний луч соседствует с первым),
    поэтому его обрабатывают одномерные свёртки с циклическим дополнением:
    сеть одинаково видит препятствие спереди и сзади и не имеет «шва».
    """

    def __init__(self, num_rays: int, hidden: int = 128) -> None:
        super().__init__()
        self.num_rays = num_rays
        self.hidden = hidden
        self.lidar_net = nn.Sequential(
            _layer(nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2, padding_mode="circular")),
            nn.ReLU(),
            _layer(nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, padding_mode="circular")),
            nn.ReLU(),
            _layer(nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode="circular")),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.lidar_net(torch.zeros(1, 1, num_rays)).shape[1]
        self.trunk = nn.Sequential(
            _layer(nn.Linear(conv_out + PROPRIO_FEATURES, hidden)),
            nn.Tanh(),
            _layer(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        # Малый масштаб выходного слоя политики: в начале обучения действия близки к нулю,
        # и исследование определяется шумом, а не случайными весами.
        self.actor = _layer(nn.Linear(hidden, 2), gain=0.01)
        self.critic = _layer(nn.Linear(hidden, 1), gain=1.0)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def features(self, scan: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat((self.lidar_net(scan.unsqueeze(1)), proprio), dim=1))

    def distribution(self, hidden: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(hidden), self.log_std.exp())

    def act(self, scan: torch.Tensor, proprio: torch.Tensor, deterministic: bool = False):
        """Действие, логарифм его вероятности и оценка ценности состояния."""
        hidden = self.features(scan, proprio)
        dist = self.distribution(hidden)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(dim=1), self.critic(hidden).squeeze(1)

    def evaluate(self, scan: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor):
        """Для обновления PPO: логарифм вероятности данного действия, энтропия, ценность."""
        hidden = self.features(scan, proprio)
        dist = self.distribution(hidden)
        return dist.log_prob(action).sum(dim=1), dist.entropy().sum(dim=1), self.critic(hidden).squeeze(1)

    def value(self, scan: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(scan, proprio)).squeeze(1)


def save_checkpoint(path: str, model: ActorCritic, sim_overrides: Mapping[str, Any], world: Mapping[str, Any],
                    extra: Mapping[str, Any] | None = None) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "num_rays": model.num_rays,
            "hidden": model.hidden,
            "sim": dict(sim_overrides),   # настройки симулятора, при которых шло обучение
            "world": dict(world),         # параметры лабиринтов
            "extra": dict(extra or {}),
        },
        path,
    )


def load_checkpoint(path: str) -> tuple[ActorCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic(checkpoint["num_rays"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def make_mazes(count: int, first_seed: int, world: Mapping[str, Any]) -> list[rs.World]:
    """Набор случайных лабиринтов: у каждого свой seed генератора."""
    return [
        rs.World.maze(cols=world["cols"], rows=world["rows"], corridor=world["corridor"], wall=0.2, cell_size=0.05,
                      seed=first_seed + i, loop_prob=world["loop_prob"])
        for i in range(count)
    ]


@torch.no_grad()
def evaluate_policy(model: ActorCritic, worlds: list[rs.World], sim_overrides: Mapping[str, Any], episodes: int,
                    seed: int = 12345, num_envs: int = 64, deterministic: bool = True) -> dict[str, float]:
    """Прогон политики на ``episodes`` эпизодах (поровну на каждую среду). Возвращает метрики.

    SPL (Success weighted by Path Length) — доля успехов, взвешенная отношением
    длины кратчайшего пути к фактически пройденному: 1 означает, что робот
    всегда доезжает, причём оптимальным маршрутом.
    """
    config = make_sim_config(sim_overrides)
    env = rs.RoverVecEnv(num_envs, worlds, config, action_mode="twist")
    obs, _ = env.reset(seed)
    dt = config.control_dt
    shortest = np.array([env.vec.env(i).shortest_path_length for i in range(num_envs)])
    travelled = np.zeros(num_envs)
    steps = np.zeros(num_envs, dtype=np.int64)
    # Каждая среда отыгрывает заранее заданное число эпизодов. Брать «первые N
    # завершившихся» нельзя: короткие эпизоды (столкновения) заканчиваются раньше
    # длинных и попали бы в выборку чаще — оценка успеха оказалась бы занижена.
    quota = np.full(num_envs, episodes // num_envs, dtype=np.int64)
    quota[: episodes % num_envs] += 1
    done_count = successes = collisions = 0
    spl = total_steps = 0.0
    while quota.any():
        scan, features = encode_observation(obs["lidar"], obs["proprio"], env.max_linear_speed, env.max_angular_speed)
        action, _, _ = model.act(scan, features, deterministic)
        obs, _, terminated, truncated, info = env.step(to_env_action(action))
        travelled += np.abs(obs["proprio"][:, OBS.ODOM_V]) * dt
        steps += 1
        for i in np.nonzero(terminated | truncated)[0]:
            if quota[i] > 0:
                quota[i] -= 1
                done_count += 1
                # Исход эпизода один: успех важнее столкновения (если оба флага
                # пришли на одном шаге), всё остальное — «не успел».
                success = bool(info["is_success"][i])
                successes += success
                collisions += (not success) and bool(info["collision"][i])
                total_steps += steps[i]
                if success:
                    spl += shortest[i] / max(shortest[i], travelled[i])
            # Среда уже перезапущена: запоминаем кратчайший путь нового эпизода.
            shortest[i] = env.vec.env(int(i)).shortest_path_length
            travelled[i] = 0.0
            steps[i] = 0
    return {
        "episodes": done_count,
        "success_rate": successes / done_count,
        "collision_rate": collisions / done_count,
        "timeout_rate": (done_count - successes - collisions) / done_count,
        "spl": spl / done_count,
        "mean_steps": total_steps / done_count,
    }
