"""Среда Gymnasium поверх симулятора."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ._core import Renderer, ShmClient, SimConfig, Simulator, WebViewer, World
from ._core import obs as OBS
from .config import make_config, make_world


class RoverEnv(gym.Env):
    """Навигация ровера к цели по лидару.

    **Действие** — ``Box(-1, 1, (2,))``:

    * ``action_mode="twist"``  — (линейная, угловая) скорости в долях максимума;
    * ``action_mode="wheels"`` — уставки (левого, правого) колёс в долях максимума.

    **Наблюдение**:

    * ``obs_mode="dict"`` — ``{"lidar": [num_rays], "proprio": [obs.DIM]}``;
    * ``obs_mode="flat"`` — один вектор: дальности лидара, затем proprio.

    Состав ``proprio`` (IMU, одометрия, направление на цель) описан индексами
    ``roversim.obs``. Награда и условия завершения задаются ``config.task``.

    **Копирование.** По умолчанию наблюдения — независимые копии, как того ждут
    буферы воспроизведения и обёртки. С ``zero_copy=True`` в режиме ``dict``
    возвращаются виды прямо в память симулятора: быстрее, но их содержимое
    меняется при следующем ``step()``, поэтому хранить их нельзя.

    **Где работает симулятор.** Обычно — в этом же процессе. Если задан
    ``shm_name``, среда подключается к запущенному ``roversim_server`` через
    разделяемую память; мир и конфигурацию тогда определяет сервер.

    **Отрисовка**: ``render_mode`` = ``"rgb_array"`` (кадр numpy), ``"ansi"``
    (текст для терминала) или ``"web"`` (живая картинка в браузере на ``web_port``).
    """

    metadata = {"render_modes": ["rgb_array", "ansi", "web"], "render_fps": 20}

    def __init__(
        self,
        world: World | str = "maze",
        config: SimConfig | Mapping[str, Any] | None = None,
        *,
        world_seed: int = 0,
        action_mode: str = "twist",
        obs_mode: str = "dict",
        zero_copy: bool = False,
        shm_name: str | None = None,
        render_mode: str | None = None,
        render_size: int = 512,
        web_port: int = 8765,
    ) -> None:
        if action_mode not in ("twist", "wheels"):
            raise ValueError("action_mode must be 'twist' or 'wheels'")
        if obs_mode not in ("dict", "flat"):
            raise ValueError("obs_mode must be 'dict' or 'flat'")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")

        if shm_name is not None:
            if render_mode is not None:
                raise ValueError("with shm_name the picture is produced by the server (roversim_server --viz ...)")
            self._sim: Simulator | ShmClient = ShmClient(shm_name)
        else:
            if isinstance(world, str):
                world = make_world(world, world_seed)
            if not isinstance(config, SimConfig):
                config = make_config(config)
            self._sim = Simulator(config, world)

        sim = self._sim
        self._twist = action_mode == "twist"
        self._flat = obs_mode == "flat"
        self._zero_copy = zero_copy
        self._wheel_radius = sim.wheel_radius
        self._track_width = sim.track_width
        self._max_wheel_speed = sim.max_wheel_speed
        self.max_linear_speed = self._wheel_radius * self._max_wheel_speed
        self.max_angular_speed = 2.0 * self.max_linear_speed / self._track_width

        # Виды на буферы симулятора создаются один раз: шаг не выделяет память под них.
        self._lidar = sim.lidar
        self._proprio = sim.proprio
        self._state = sim.state

        num_rays = sim.num_rays
        range_max = np.float32(sim.lidar_range_max)
        lidar_space = spaces.Box(0.0, range_max, (num_rays,), np.float32)
        proprio_space = spaces.Box(-np.inf, np.inf, (OBS.DIM,), np.float32)
        if self._flat:
            low = np.concatenate([lidar_space.low, proprio_space.low])
            high = np.concatenate([lidar_space.high, proprio_space.high])
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.observation_space = spaces.Dict({"lidar": lidar_space, "proprio": proprio_space})
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.render_mode = render_mode
        self._render_size = render_size
        self._renderer: Renderer | None = None
        self._viewer = WebViewer(web_port) if render_mode == "web" else None

    # ------------------------------------------------------------ Gymnasium API

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        """Новый эпизод.

        ``options`` может задать ``{"start": (x, y, theta), "goal": (x, y)}``;
        иначе старт и цель случайны. Достаточно один раз передать ``seed``,
        чтобы вся последовательность эпизодов стала воспроизводимой.
        """
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**63 - 1))
        options = options or {}
        self._sim.reset(episode_seed, options.get("start"), options.get("goal"))
        if self._viewer is not None:
            self._viewer.publish(self._sim, True)
        return self._observation(), self._info(False, False)

    def step(self, action):
        left, right = self._wheel_speeds(action)
        result = self._sim.step(left, right)
        if self._viewer is not None:
            self._viewer.publish(self._sim, result.terminated or result.truncated)
        info = self._info(result.success, result.collision)
        return self._observation(), result.reward, result.terminated, result.truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "web":
            self._viewer.publish(self._sim, True)
            return None
        if self._renderer is None:
            # Для терминала кадр маленький: одно знакоместо вмещает 1 × 2 пикселя.
            size = self._render_size if self.render_mode == "rgb_array" else min(self._render_size, 96)
            self._renderer = Renderer(size, size)
        if self.render_mode == "rgb_array":
            return self._renderer.render(self._sim)
        return self._renderer.render_ansi(self._sim)

    def close(self) -> None:
        self._viewer = None
        self._renderer = None

    # --------------------------------------------------------------- доступ к ядру

    @property
    def sim(self) -> Simulator | ShmClient:
        """Симулятор (или клиент разделяемой памяти), на котором работает среда."""
        return self._sim

    @property
    def state(self) -> np.ndarray:
        """Истинное состояние (индексы — ``roversim.gt``): вид без копирования."""
        return self._state

    @property
    def lidar_angles(self) -> np.ndarray:
        """Углы лучей лидара в системе координат сенсора [рад]."""
        return self._sim.lidar_angles

    # ------------------------------------------------------------------ внутреннее

    def _wheel_speeds(self, action) -> tuple[float, float]:
        a0 = min(1.0, max(-1.0, float(action[0])))
        a1 = min(1.0, max(-1.0, float(action[1])))
        if not self._twist:
            return a0 * self._max_wheel_speed, a1 * self._max_wheel_speed
        # Обратная кинематика дифференциального привода. Если уставка колеса
        # выходит за предел, обе уменьшаются одинаково: кривизна дуги сохраняется.
        v = a0 * self.max_linear_speed
        w = a1 * self.max_angular_speed
        left = (v - 0.5 * w * self._track_width) / self._wheel_radius
        right = (v + 0.5 * w * self._track_width) / self._wheel_radius
        peak = max(abs(left), abs(right))
        if peak > self._max_wheel_speed:
            scale = self._max_wheel_speed / peak
            left *= scale
            right *= scale
        return left, right

    def _observation(self):
        if self._flat:
            return np.concatenate((self._lidar, self._proprio))
        if self._zero_copy:
            return {"lidar": self._lidar, "proprio": self._proprio}
        return {"lidar": self._lidar.copy(), "proprio": self._proprio.copy()}

    def _info(self, success: bool, collision: bool) -> dict[str, Any]:
        return {"is_success": success, "collision": collision, "time": self._sim.time}


gym.register(id="roversim/Nav-v0", entry_point="roversim.env:RoverEnv")
