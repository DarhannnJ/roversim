"""Тесты Python-интерфейса.

Запуск (из корня проекта, после сборки)::

    PYTHONPATH=build/python python -m pytest -q tests/test_python.py

Тест обмена с ``roversim_server`` выполняется, если задана переменная окружения
``ROVERSIM_SERVER`` с путём к исполняемому файлу (CTest задаёт её сам).
"""

from __future__ import annotations

import os
import subprocess
import threading
import urllib.request

import numpy as np
import pytest

import roversim as rs
from roversim import gt, nav, obs


def make_sim(**overrides) -> rs.Simulator:
    return rs.Simulator(rs.make_config(overrides), rs.make_world("maze", seed=1))


def wander(step: int) -> tuple[float, float]:
    phase = (step // 40) % 5
    return ((14.0, 16.0, 6.0, 16.0, -6.0)[phase], (14.0, 6.0, 16.0, 16.0, 6.0)[phase])


# ------------------------------------------------------------------------ ядро


def test_observations_are_zero_copy_views():
    sim = make_sim()
    sim.reset(seed=3)
    lidar, proprio, state = sim.lidar, sim.proprio, sim.state
    assert lidar.shape == (360,) and lidar.dtype == np.float32
    assert proprio.shape == (obs.DIM,) and proprio.dtype == np.float32
    assert state.shape == (gt.DIM,) and state.dtype == np.float64
    # Вид без копирования: память принадлежит симулятору и защищена от записи.
    assert lidar.base is sim and not lidar.flags.owndata and not lidar.flags.writeable
    with pytest.raises(ValueError):
        lidar[0] = 1.0
    # Повторный доступ даёт тот же буфер, а шаг меняет его содержимое на месте.
    assert sim.lidar.ctypes.data == lidar.ctypes.data
    before = lidar.copy()
    sim.step(10.0, 12.0)
    assert not np.array_equal(before, lidar)
    # Массив удерживает симулятор: после удаления имени память остаётся доступной.
    del sim
    assert np.isfinite(lidar).all() and lidar.min() > 0.0


def test_simulation_is_deterministic():
    def run(seed: int) -> np.ndarray:
        sim = make_sim()
        sim.reset(seed=seed)
        rows = []
        for step in range(300):
            result = sim.step(*wander(step))
            rows.append(np.concatenate([sim.lidar, sim.proprio, sim.state, [result.reward]]))
            if result.terminated or result.truncated:
                sim.reset(seed=seed + step)
        return np.array(rows)

    assert np.array_equal(run(5), run(5))
    assert not np.array_equal(run(5), run(6))


def test_config_overrides_and_validation():
    config = rs.make_config({"lidar.num_rays": 180, "robot.mass": 8.0}, task__max_steps=50, substeps=5)
    assert config.lidar.num_rays == 180 and config.robot.mass == 8.0
    assert config.task.max_steps == 50 and config.control_dt == pytest.approx(0.025)
    with pytest.raises(KeyError):
        rs.make_config({"lidar.num_ray": 1})          # опечатка в имени поля
    with pytest.raises(ValueError):
        rs.Simulator(rs.make_config({"robot.mass": -1.0}), rs.make_world("arena"))
    sim = make_sim()
    with pytest.raises(ValueError):
        sim.reset(1, start=(0.01, 0.01, 0.0), goal=(3.0, 3.0))   # старт внутри стены
    with pytest.raises(ValueError):
        sim.reset(1, start=(3.0, 3.0, 0.0))                       # старт без цели


def test_world_construction_and_queries():
    grid = np.zeros((40, 60), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    grid[10:30, 30] = 1                                  # перегородка x = 3.0 … 3.1 м
    world = rs.World.from_grid(grid, cell_size=0.1)
    assert (world.width, world.height) == (60, 40) and world.bounds == [0.0, 0.0, 6.0, 4.0]
    assert np.array_equal(world.grid, grid) and not world.grid.flags.writeable
    assert world.raycast(1.0, 2.0, 0.0, 10.0) == pytest.approx(2.0)          # до перегородки
    assert world.raycast(1.0, 0.5, 0.0, 10.0) == pytest.approx(4.9)          # мимо неё, до стены
    assert world.is_free(1.0, 2.0, 0.3) and not world.is_free(2.9, 2.0, 0.3)

    built = rs.World(60, 40, 0.1)
    built.fill_box(0.0, 0.0, 6.0, 0.1)
    built.add_circle(3.0, 2.0, 0.5)
    built.add_box(4.0, 1.0, 4.5, 3.0)
    built.finalize()
    assert built.raycast(1.0, 2.0, 0.0, 10.0) == pytest.approx(1.5)          # до круга
    assert built.circles == [[3.0, 2.0, 0.5]] and len(built.boxes) == 1

    ascii_world = rs.World.from_ascii(["#####", "#...#", "#####"], cell_size=0.5)
    assert ascii_world.occupied(0, 1) and not ascii_world.occupied(2, 1)
    assert rs.make_world("clutter", seed=3).circles != rs.make_world("clutter", seed=4).circles


def test_moving_obstacles_and_path_hint():
    sim = rs.Simulator(rs.make_config({"obstacles.count": 3, "task.terminate_on_collision": False}),
                       rs.make_world("arena"))
    sim.reset(seed=2)
    obstacles = sim.obstacles
    assert obstacles.shape == (3, 5) and np.all(obstacles[:, 4] == pytest.approx(0.2))
    start = obstacles[:, :2].copy()
    for _ in range(100):
        sim.step(0.0, 0.0)
    assert np.all(np.linalg.norm(sim.obstacles[:, :2] - start, axis=1) > 0.3)   # препятствия едут

    # На пустой арене подсказка маршрута смотрит прямо на цель.
    empty = rs.Simulator(rs.make_config(), rs.make_world("arena"))
    empty.reset(seed=1)
    p = empty.proprio
    assert p[obs.PATH_COS] == pytest.approx(p[obs.GOAL_COS], abs=0.05)
    assert p[obs.PATH_SIN] == pytest.approx(p[obs.GOAL_SIN], abs=0.05)


# -------------------------------------------------------------------- Gymnasium


# Одометрическая поза и акселерометр не ограничены по смыслу, поэтому границы
# пространства наблюдений бесконечны; проверка Gymnasium об этом лишь предупреждает.
@pytest.mark.filterwarnings("ignore:.*Box observation space:UserWarning")
def test_gym_env_passes_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(rs.RoverEnv(config={"lidar.num_rays": 90}), skip_render_check=True)
    check_env(rs.RoverEnv(config={"lidar.num_rays": 90}, obs_mode="flat", action_mode="wheels"), skip_render_check=True)


def test_gym_env_seeding_and_copy_semantics():
    import gymnasium

    env = gymnasium.make("roversim/Nav-v0", config={"lidar.num_rays": 90})
    first, _ = env.reset(seed=11)
    again, _ = env.reset(seed=11)
    assert np.array_equal(first["lidar"], again["lidar"])
    # Один seed задаёт всю последовательность эпизодов: второй эпизод тоже совпадает.
    second_a, _ = env.reset()
    env.reset(seed=11)
    second_b, _ = env.reset()
    assert np.array_equal(second_a["lidar"], second_b["lidar"]) and not np.array_equal(first["lidar"], second_a["lidar"])

    # По умолчанию наблюдения — независимые копии.
    observation, _ = env.reset(seed=1)
    kept = observation["lidar"].copy()
    env.step(env.action_space.sample())
    assert np.array_equal(observation["lidar"], kept)
    assert not np.shares_memory(observation["lidar"], env.unwrapped.sim.lidar)

    # zero_copy=True: наблюдение — вид на память симулятора.
    fast = rs.RoverEnv(config={"lidar.num_rays": 90}, zero_copy=True)
    observation, _ = fast.reset(seed=1)
    assert np.shares_memory(observation["lidar"], fast.sim.lidar)

    flat = rs.RoverEnv(config={"lidar.num_rays": 90}, obs_mode="flat")
    vector, _ = flat.reset(seed=1)
    assert vector.shape == (90 + obs.DIM,) and flat.observation_space.contains(vector)


def test_gym_env_explicit_reset_and_actions():
    env = rs.RoverEnv("arena", {"lidar.num_rays": 90})
    observation, info = env.reset(seed=1, options={"start": (2.0, 2.0, 0.0), "goal": (5.0, 2.0)})
    assert env.state[gt.X] == pytest.approx(2.0) and env.state[gt.GOAL_X] == pytest.approx(5.0)
    assert observation["proprio"][obs.GOAL_DISTANCE] == pytest.approx(3.0, abs=1e-5)
    # Полный вперёд: робот едет к цели и получает положительную награду за прогресс.
    total, done = 0.0, False
    while not done:
        observation, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        total += reward
        done = terminated or truncated
    assert info["is_success"] and not info["collision"] and total > 10.0
    # Поворот на месте в режиме twist: колёса крутятся в разные стороны.
    env.reset(seed=1, options={"start": (2.0, 2.0, 0.0), "goal": (5.0, 2.0)})
    for _ in range(10):
        env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert env.state[gt.WHEEL_SPEED_LEFT] < -1.0 and env.state[gt.WHEEL_SPEED_RIGHT] > 1.0
    assert abs(env.state[gt.X] - 2.0) < 1e-3


def test_rendering():
    env = rs.RoverEnv(config={"lidar.num_rays": 90}, render_mode="rgb_array", render_size=200)
    env.reset(seed=1)
    frame = env.render()
    assert frame.shape == (200, 200, 3) and frame.dtype == np.uint8 and len(np.unique(frame)) > 10
    text = rs.RoverEnv(config={"lidar.num_rays": 90}, render_mode="ansi")
    text.reset(seed=1)
    assert "▀" in text.render()


def test_web_viewer_serves_page():
    viewer = rs.WebViewer(0)   # любой свободный порт
    page = urllib.request.urlopen(f"http://127.0.0.1:{viewer.port}/", timeout=5).read().decode()
    assert "<canvas" in page and viewer.teleop() is None


# ----------------------------------------------------------------- пакет сред


def test_vec_env_shapes_and_thread_invariance():
    def rollout(threads: int):
        env = rs.RoverVecEnv(12, "maze", {"lidar.num_rays": 90, "task.max_steps": 40}, num_worlds=3,
                             num_threads=threads, zero_copy=False)
        observation, _ = env.reset(seed=4)
        rng = np.random.default_rng(0)
        trace = [observation["lidar"]]
        done_total = 0
        for _ in range(120):
            observation, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, (12, 2)))
            trace += [observation["lidar"], observation["proprio"], reward]
            done_total += int((terminated | truncated).sum())
        return trace, done_total, (observation, reward, terminated, truncated, info)

    single, done_single, last = rollout(1)
    multi, done_multi, _ = rollout(4)
    assert all(np.array_equal(a, b) for a, b in zip(single, multi))   # результат не зависит от числа потоков
    assert done_single == done_multi > 12                              # эпизоды завершались и перезапускались
    observation, reward, terminated, truncated, info = last
    assert observation["lidar"].shape == (12, 90) and observation["proprio"].shape == (12, obs.DIM)
    assert reward.shape == (12,) and reward.dtype == np.float32
    assert terminated.dtype == np.bool_ and info["is_success"].shape == (12,)


def test_vec_env_zero_copy_into_torch():
    torch = pytest.importorskip("torch")
    import warnings

    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    env = rs.RoverVecEnv(8, "arena", {"lidar.num_rays": 90})
    observation, _ = env.reset(seed=1)
    tensor = torch.from_numpy(observation["lidar"])
    assert tensor.shape == (8, 90) and tensor.data_ptr() == observation["lidar"].ctypes.data
    before = tensor.clone()
    env.step(np.ones((8, 2)))
    assert not torch.equal(before, tensor)   # тензор видит новый скан без копирования


# ------------------------------------------------------------------- навигация


def test_occupancy_mapping_and_planner_from_python():
    world = rs.make_world("maze", seed=1)
    planner = nav.GridPlanner()
    planner.set_map(world)                       # планирование по истинной карте
    sim = rs.Simulator(rs.make_config(), world)
    sim.reset(seed=2)
    path = planner.plan(sim.pose[:2], sim.goal)
    assert path is not None and path.shape[1] == 2
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    assert 0.85 * sim.shortest_path_length - 0.5 < length < 1.25 * sim.shortest_path_length + 0.5
    assert all(world.is_free(float(x), float(y), 0.2) for x, y in path)
    assert planner.clearance(*sim.pose[:2]) > 0.2

    astar_config = nav.PlannerConfig()
    astar_config.algorithm = nav.Algorithm.ASTAR
    astar_config.smooth = False
    astar = nav.GridPlanner(astar_config)
    astar.set_map(world)
    grid_path = astar.plan(sim.pose[:2], sim.goal)
    assert len(grid_path) > len(path)            # путь Theta* состоит из меньшего числа отрезков

    # Карта занятости по сканам из известной позы.
    scan = nav.scan_geometry(sim)
    grid = nav.OccupancyGrid(nav.MapConfig(), center=tuple(sim.pose[:2]))
    for _ in range(5):
        grid.integrate_scan(sim.pose, sim.lidar, scan)
    states = grid.states()
    assert grid.log_odds.shape == states.shape == (320, 320) and not grid.log_odds.flags.writeable
    assert (states == 0).sum() > 500 and (states == 1).sum() > 50 and (states == 2).sum() > 50_000
    probability = grid.probability()
    assert probability.min() < 0.2 and probability.max() > 0.9


def test_navigation_agent_reaches_goal_with_slam():
    sim = rs.Simulator(rs.make_config({"task.max_steps": 3000, "task.terminate_on_collision": False}),
                       rs.make_world("maze", seed=103))
    sim.reset(seed=3)
    agent = rs.NavigationAgent(sim)
    agent.set_goal(*sim.goal)
    contacts = 0
    while True:
        result = sim.step(*agent.act())
        contacts += result.collision
        if result.terminated or result.truncated:
            break
    navigator = agent.navigator
    assert result.success and contacts == 0
    # Оценка позы SLAM остаётся рядом с истиной.
    assert np.linalg.norm(np.array(navigator.pose[:2]) - np.array(sim.pose[:2])) < 0.1
    assert navigator.replans >= 1 and navigator.map.scans == sim.step_count
    image = rs.render_navigation(navigator, truth_pose=sim.pose)
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8


def test_dwa_planner_from_python():
    dwa = nav.DwaPlanner()
    free = dwa.compute(np.zeros((0, 2)), target=(2.0, 0.0), v=0.0, w=0.0)
    assert free.feasible and free.v > 0.0 and abs(free.w) < 0.3
    wall = np.array([[0.45, y] for y in np.arange(-1.5, 1.5, 0.03)])
    blocked = dwa.compute(wall, target=(2.0, 0.0), v=0.5, w=0.0)
    assert blocked.v < 0.5
    retreat = dwa.compute(np.zeros((0, 2)), target=(2.0, 0.0), v=0.0, w=0.0,
                          moving=[(1.0, 0.0, -0.5, 0.0, 0.2)], allow_reverse=True)
    assert retreat.v < 0.0 and dwa.trajectory.shape[1] == 2


# ------------------------------------------------------------ разделяемая память


def test_shared_memory_matches_in_process_simulation():
    name = f"/roversim_pytest_{os.getpid()}"
    config = rs.make_config({"lidar.num_rays": 120})
    world = rs.make_world("maze", seed=1)
    server = rs.ShmServer(name, config, world)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            if server.serve_once(50) == 3:   # команда close
                break

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        client = rs.ShmClient(name)
        local = rs.Simulator(config, world)
        assert client.num_rays == 120 and client.control_dt == pytest.approx(config.control_dt)
        client.reset(7)
        local.reset(7)
        lidar = client.lidar                    # вид прямо в разделяемую память
        assert not lidar.flags.owndata and np.array_equal(lidar, local.lidar)
        for step in range(200):
            remote = client.step(*wander(step))
            expected = local.step(*wander(step))
            assert remote.reward == expected.reward and remote.terminated == expected.terminated
            assert np.array_equal(lidar, local.lidar) and np.array_equal(client.state, local.state)
            if remote.terminated or remote.truncated:
                client.reset(100 + step)
                local.reset(100 + step)
        # Диагностика и настройка ожидания: сервер сообщает время выполнения команды,
        # время активного ожидания меняется на ходу и на результат не влияет.
        assert 0.0 < server.last_command_seconds < 1.0
        assert server.spin_time == pytest.approx(200e-6) and client.spin_time == pytest.approx(200e-6)
        server.spin_time = 0.0
        client.spin_time = 0.0
        for step in range(20):
            assert client.step(*wander(step)).reward == local.step(*wander(step)).reward
        server.spin_time = 1e-3
        assert server.spin_time == pytest.approx(1e-3)
        with pytest.raises(ValueError):
            client.spin_time = -1.0
        observer = rs.ShmClient(name, read_only=True)
        snapshot = observer.snapshot()
        assert snapshot is not None and np.array_equal(snapshot[0], local.lidar)
        with pytest.raises(RuntimeError):
            client.reset(1, start=(0.01, 0.01, 0.0), goal=(3.0, 3.0))   # сервер отклоняет, но продолжает работать
        client.reset(9)
        local.reset(9)
        assert np.array_equal(client.lidar, local.lidar)
        client.close_server()
    finally:
        stop.set()
        thread.join()


@pytest.mark.skipif(not os.environ.get("ROVERSIM_SERVER"), reason="ROVERSIM_SERVER is not set")
def test_gym_env_over_server_process():
    name = f"/roversim_pytest_srv_{os.getpid()}"
    server = subprocess.Popen([os.environ["ROVERSIM_SERVER"], "--shm", name, "--world", "maze", "--world-seed", "1", "--rays", "360"])
    try:
        remote = rs.RoverEnv(shm_name=name)
        # Тот же мир и те же настройки, что у сервера по умолчанию, но в этом процессе.
        local = rs.RoverEnv(rs.World.maze(5, 5, 1.2, 0.2, 0.05, 1, 0.15))
        a, _ = remote.reset(seed=5)
        b, _ = local.reset(seed=5)
        assert np.array_equal(a["lidar"], b["lidar"])
        rng = np.random.default_rng(0)
        for _ in range(100):
            action = rng.uniform(-1, 1, 2).astype(np.float32)
            a, reward_a, term_a, trunc_a, _ = remote.step(action)
            b, reward_b, term_b, trunc_b, _ = local.step(action)
            assert np.array_equal(a["lidar"], b["lidar"]) and reward_a == reward_b
            assert (term_a, trunc_a) == (term_b, trunc_b)
            if term_a or trunc_a:
                a, _ = remote.reset()
                b, _ = local.reset()
                assert np.array_equal(a["proprio"], b["proprio"])
        remote.sim.close_server()
    finally:
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.terminate()
