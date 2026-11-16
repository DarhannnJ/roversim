"""Пакетная среда: num_envs симуляторов шагают параллельно в потоках C++."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import SimConfig, VecSimulator, World
from ._core import obs as OBS
from .config import make_config, make_world


class RoverVecEnv:
    """Пакет сред для обучения с подкреплением.

    Интерфейс повторяет векторные среды Gymnasium::

        env = RoverVecEnv(256, world="maze", num_worlds=16)
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(actions)   # actions: [N, 2] в [-1, 1]

    * ``obs["lidar"]`` — ``float32 [N, num_rays]``, ``obs["proprio"]`` — ``float32 [N, obs.DIM]``;
    * ``reward`` — ``float32 [N]``; ``terminated``, ``truncated`` — ``bool [N]``;
    * ``info`` — ``{"is_success": bool [N], "collision": bool [N]}``.

    **Без копирования.** По умолчанию все возвращаемые массивы — виды на общие
    буферы симулятора: их можно сразу превращать в тензоры
    (``torch.from_numpy(obs["lidar"])``), но содержимое меняется при следующем
    ``step()``. Если данные нужно хранить между шагами, копируйте их или
    создайте среду с ``zero_copy=False``.

    **Автоперезапуск.** Завершившаяся среда перезапускается в том же вызове
    ``step()``: её наблюдение уже относится к новому эпизоду, а награда и флаги —
    к последнему шагу завершённого.

    Параметры ``world`` и ``num_worlds``: строка (``"maze"``, ``"arena"``,
    ``"clutter"``) порождает ``num_worlds`` разных миров, между которыми
    распределяются среды; можно передать и готовый ``World`` или их список.
    """

    def __init__(
        self,
        num_envs: int,
        world: World | Sequence[World] | str = "maze",
        config: SimConfig | Mapping[str, Any] | None = None,
        *,
        world_seed: int = 0,
        num_worlds: int = 1,
        action_mode: str = "twist",
        num_threads: int = 0,
        zero_copy: bool = True,
    ) -> None:
        if action_mode not in ("twist", "wheels"):
            raise ValueError("action_mode must be 'twist' or 'wheels'")
        if isinstance(world, str):
            worlds: Sequence[World] = [make_world(world, world_seed + i) for i in range(max(1, num_worlds))]
        elif isinstance(world, World):
            worlds = [world]
        else:
            worlds = list(world)
        if not isinstance(config, SimConfig):
            config = make_config(config)

        self._vec = VecSimulator(config, worlds, num_envs, num_threads)
        self.num_envs = num_envs
        self.num_rays = self._vec.num_rays
        self.proprio_dim = OBS.DIM
        self.config = config
        self._twist = action_mode == "twist"
        self._zero_copy = zero_copy

        robot = config.robot
        self._wheel_radius = robot.wheel_radius
        self._track_width = robot.track_width
        self._max_wheel_speed = robot.max_wheel_speed
        self.max_linear_speed = robot.wheel_radius * robot.max_wheel_speed
        self.max_angular_speed = 2.0 * self.max_linear_speed / robot.track_width

        # Виды на пакетные буферы и буфер действий создаются один раз.
        vec = self._vec
        self._lidar, self._proprio, self._state = vec.lidar, vec.proprio, vec.state
        self._rewards, self._terminated, self._truncated = vec.rewards, vec.terminated, vec.truncated
        self._collision, self._success = vec.collision, vec.success
        self._wheels = np.zeros((num_envs, 2), dtype=np.float64)

    def reset(self, seed: int = 0):
        self._vec.reset(seed)
        return self._observation(), {}

    def step(self, actions):
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(self.num_envs, 2), -1.0, 1.0)
        wheels = self._wheels
        if self._twist:
            # Обратная кинематика; при насыщении обе уставки уменьшаются одинаково.
            v = a[:, 0] * self.max_linear_speed
            half_turn = 0.5 * self._track_width * a[:, 1] * self.max_angular_speed
            wheels[:, 0] = (v - half_turn) / self._wheel_radius
            wheels[:, 1] = (v + half_turn) / self._wheel_radius
            peak = np.abs(wheels).max(axis=1, keepdims=True)
            wheels /= np.maximum(1.0, peak / self._max_wheel_speed)
        else:
            np.multiply(a, self._max_wheel_speed, out=wheels)
        self._vec.step(wheels)

        info = {"is_success": self._success, "collision": self._collision}
        if self._zero_copy:
            return self._observation(), self._rewards, self._terminated, self._truncated, info
        info = {key: value.copy() for key, value in info.items()}
        return self._observation(), self._rewards.copy(), self._terminated.copy(), self._truncated.copy(), info

    def close(self) -> None:
        """Метод интерфейса Gymnasium. Закрывать нечего: потоки и буферы принадлежат
        объекту ``VecSimulator`` и освобождаются вместе с ним."""

    @property
    def vec(self) -> VecSimulator:
        """Пакетный симулятор ядра."""
        return self._vec

    @property
    def state(self) -> np.ndarray:
        """Истинное состояние ``float64 [N, gt.DIM]`` — вид без копирования."""
        return self._state

    def _observation(self) -> dict[str, np.ndarray]:
        if self._zero_copy:
            return {"lidar": self._lidar, "proprio": self._proprio}
        return {"lidar": self._lidar.copy(), "proprio": self._proprio.copy()}
