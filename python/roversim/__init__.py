"""roversim — лёгкий детерминированный симулятор ровера с дифференциальным приводом.

Быстрый старт::

    import roversim as rs

    world = rs.World.maze(cols=5, rows=5, seed=1)
    sim = rs.Simulator(rs.make_config({"lidar.num_rays": 360}), world)
    sim.reset(seed=42)
    result = sim.step(10.0, 10.0)   # уставки угловых скоростей колёс, рад/с
    scan = sim.lidar                # numpy float32 [num_rays], без копирования

Уровни интерфейса:

* ``Simulator`` / ``VecSimulator`` — ядро: одна среда или пакет сред;
* ``RoverEnv`` — среда Gymnasium (нужен пакет ``gymnasium``);
* ``RoverVecEnv`` — пакетная среда для обучения с подкреплением;
* ``ShmClient`` — управление симулятором, работающим в другом процессе;
* ``nav`` и ``NavigationAgent`` — автономная навигация: SLAM, Theta*, DWA.
"""

from ._core import (
    ImuConfig,
    LidarConfig,
    OdomConfig,
    Renderer,
    RobotParams,
    ShmClient,
    ShmServer,
    SimConfig,
    Simulator,
    StepResult,
    TaskConfig,
    VecSimulator,
    WebViewer,
    World,
    __version__,
    gt,
    nav,
    obs,
)
from .config import make_config, make_world
from .navigation import NavigationAgent, make_navigator, render_navigation
from .vec_env import RoverVecEnv

__all__ = [
    "ImuConfig",
    "LidarConfig",
    "NavigationAgent",
    "OdomConfig",
    "Renderer",
    "RobotParams",
    "RoverVecEnv",
    "ShmClient",
    "ShmServer",
    "SimConfig",
    "Simulator",
    "StepResult",
    "TaskConfig",
    "VecSimulator",
    "WebViewer",
    "World",
    "__version__",
    "gt",
    "make_config",
    "make_navigator",
    "make_world",
    "nav",
    "obs",
    "render_navigation",
]

try:
    from .env import RoverEnv

    __all__.append("RoverEnv")
except ModuleNotFoundError as error:  # ядро работает и без gymnasium
    if error.name != "gymnasium":
        raise
