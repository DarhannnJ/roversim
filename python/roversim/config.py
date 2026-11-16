"""Удобные конструкторы конфигурации и стандартных миров."""

from __future__ import annotations

from typing import Any, Mapping

from ._core import SimConfig, World


def make_config(overrides: Mapping[str, Any] | None = None, **kwargs: Any) -> SimConfig:
    """Собирает ``SimConfig``, переопределяя отдельные поля.

    Ключи — пути через точку::

        make_config({"lidar.num_rays": 180, "robot.mass": 8.0, "substeps": 5})

    В именованных аргументах точку заменяет двойное подчёркивание::

        make_config(lidar__num_rays=180, task__max_steps=300)

    Опечатка в имени поля — ``KeyError``, а не молча проигнорированная настройка.
    """
    config = SimConfig()
    items = dict(overrides or {})
    items.update({key.replace("__", "."): value for key, value in kwargs.items()})
    for path, value in items.items():
        *parents, field = path.split(".")
        target = config
        for name in parents:
            if not hasattr(target, name):
                raise KeyError(f"unknown config section: {path!r}")
            target = getattr(target, name)
        if not hasattr(target, field):
            raise KeyError(f"unknown config field: {path!r}")
        setattr(target, field, value)
    return config


def make_world(kind: str = "maze", seed: int = 0) -> World:
    """Стандартные миры: ``"maze"``, ``"arena"`` или ``"clutter"``.

    * maze    — лабиринт 5 × 5 ячеек с проходами 1.2 м и петлями (7.2 × 7.2 м);
    * arena   — пустая комната 8 × 8 м;
    * clutter — комната 8 × 8 м со случайными препятствиями.
    """
    if kind == "maze":
        return World.maze(cols=5, rows=5, corridor=1.2, wall=0.2, cell_size=0.05, seed=seed, loop_prob=0.15)
    if kind == "arena":
        return World.arena(8.0, 8.0, cell_size=0.05)
    if kind == "clutter":
        return World.cluttered(8.0, 8.0, cell_size=0.05, num_boxes=6, num_pillars=6, seed=seed)
    raise ValueError(f"unknown world kind {kind!r}: expected 'maze', 'arena' or 'clutter'")
