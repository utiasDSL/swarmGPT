"""Simulation module for swarm_gpt.

Before we deploy the choreography to the drones, we run a simulation to check if the modified paths
from AMSwarm are collision-free and can be executed. While there is no guarantee that the
trajectories work in reality, it is a good sanity check to ensure that the drones do not crash into
each other or have to perform infeasible maneuvers.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
import PIL
from axswarm import SolverData, SolverSettings, solve
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from tqdm import tqdm

from swarm_gpt.utils import MusicManager
from swarm_gpt.utils.utils import draw_line

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swarm_gpt.utils import MusicManager


def simulate_axswarm(
    waypoints: dict[str, NDArray], settings: dict, gui: bool = False
) -> dict[int, NDArray]:
    """Run the crazyflow simulation from waypoints.

    Args:
        waypoints: The waypoints to fly to. Dictionary of drone IDs to waypoints. Each waypoint
            consists of [time, x, y, z, vx, vy, vz].
        settings: Settings for the simulation and AMSwarm.
        gui: Flag to render the simulation.

    Returns:
        A collection of data from the simulation.
    """
    # Set up the simulation
    sim = Sim(
        n_worlds=1,
        n_drones=waypoints["pos"].shape[0],
        physics=Physics.analytical,
        control=Control.state,
        freq=settings["sim_freq"],
        attitude_freq=settings["attitude_freq"],
        state_freq=settings["state_freq"],
        device="cpu",
    )
    fps = 60
    sim.max_visual_geom = 100_000

    # JIT compile the simulation
    sim.reset()
    sim.state_control(np.random.random((sim.n_worlds, sim.n_drones, 13)))
    sim.step(sim.freq // sim.control_freq)
    sim.reset()

    # Set up solver
    solver_settings = {
        k: v if not isinstance(v, list) else np.asarray(v) for k, v in settings["axswarm"].items()
    }
    solver_settings = SolverSettings(**solver_settings)
    dynamics = settings["Dynamics"]
    A, B = np.asarray(dynamics["A"]), np.asarray(dynamics["B"])
    A_prime, B_prime = np.asarray(dynamics["A_prime"]), np.asarray(dynamics["B_prime"])
    solver_data = SolverData.init(
        waypoints=waypoints,
        K=solver_settings.K,
        N=solver_settings.N,
        A=A,
        B=B,
        A_prime=A_prime,
        B_prime=B_prime,
        freq=solver_settings.freq,
        smoothness_weight=solver_settings.smoothness_weight,
        input_smoothness_weight=solver_settings.input_smoothness_weight,
        input_continuity_weight=solver_settings.input_continuity_weight,
    )
    n_steps = int(waypoints["time"][0, -1] * sim.control_freq)
    solve_every_n_steps = sim.control_freq // solver_settings.freq
    assert sim.freq % sim.control_freq == 0, (
        "control freq {sim.control_freq} must be divisible by sim.freq {sim.freq}"
    )
    assert sim.control_freq % solver_settings.freq == 0, (
        "control freq {sim.control_freq} must be divisible by amswarm freq {solver_settings.freq}"
    )

    # Set up initial states
    control = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
    pos = sim.data.states.pos.at[0, ...].set(waypoints["pos"][:, 0])
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    pos, vel = np.asarray(sim.data.states.pos[0]), np.asarray(sim.data.states.vel[0])
    states, controls, solve_times = [], [], []  # logging variables

    # Set up colours for tracking lines
    rng = np.random.default_rng(0)
    rgbas = rng.random((sim.n_drones, 4))
    rgbas[..., 3] = 1

    tstart = time.time()
    for step in tqdm(range(n_steps)):
        yield "progress", step+1, n_steps
        t = step / sim.control_freq
        if step % solve_every_n_steps == 0:
            state = np.concat((pos, vel), axis=-1)
            t_solve = time.perf_counter()
            success, _, solver_data = solve(state, t, solver_data, solver_settings)
            jax.block_until_ready(solver_data)
            solve_times.append(time.perf_counter() - t_solve)
            if not all(success):
                logger.info("Solve failed")

            solver_data = solver_data.step(solver_data)
            pos, vel = solver_data.u_pos[:, 0], solver_data.u_vel[:, 0]
            control[0, :, :3] = solver_data.u_pos[:, 0]
            control[0, :, 3:6] = solver_data.u_vel[:, 0]

            # Log inputs
            controls.append(control[0, :, :6].copy())
            states.append(control[0, :, :6].copy())

        # Run the simulation
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)

        # Render simulation with visualizations of the planned trajectories
        if ((step * fps) % sim.control_freq) < fps and gui:
            for i in range(sim.n_drones):
                draw_line(sim, solver_data.u_pos[i, :], rgba=rgbas[i % len(rgbas)])
            sim.render()
            if (dt := t - (time.time() - tstart)) > 0:
                time.sleep(dt)
    sim.close()

    sim_log = {
        "num_drones": sim.n_drones,
        "log_freq": solver_settings.freq,
        "sim_freq": sim.freq,
        "timestamps": np.arange(n_steps) / sim.control_freq,
        "states": np.array(states),
        "controls": np.array(controls),
        "waypoints": waypoints,
        "simulation_freq": sim.freq,
        "amswarm_every_n_steps": solve_every_n_steps,
        "solve_times": np.array(solve_times),
    }
    yield "result", sim_log, "placeholder"
    #return sim_log


def simulate_spline(
    splines: dict, settings: dict, t: float, music_manager: MusicManager, gui: bool
):
    """Run the simulation using splines as control reference."""
    # Setting Up Simulation
    fps = 60
    amswarm_freq = settings["axswarm"]["freq"]
    sim = Sim(
        n_worlds=1,
        n_drones=len(splines),
        physics=Physics.analytical,
        control=Control.state,
        freq=settings["sim_freq"],
        attitude_freq=settings["attitude_freq"],
        state_freq=settings["state_freq"],
        device="cpu",
    )
    sim.max_visual_geom = 100_000
    # JIT compile the simulation
    sim.reset()
    sim.state_control(np.random.random((1, sim.n_drones, 13)))
    sim.step(sim.freq // sim.control_freq)
    sim.reset()

    vel_splines = {i: [s.derivative() for s in splines[i]] for i in splines}
    assert sim.freq % sim.control_freq == 0, (
        "control freq {sim.control_freq} must be divisible by sim.freq {sim.freq}"
    )
    assert sim.control_freq % amswarm_freq == 0, (
        "control freq {sim.control_freq} must be divisible by amswarm freq {amswarm_freq}"
    )
    # Setting Up Initial States
    pos = np.array([[s(0) for s in splines[j]] for j in splines])[None, ...]
    assert pos.shape == sim.data.states.pos.shape, (
        f"Initial drone position shape mismatch ({pos.shape}) vs ({sim.data.states.pos.shape})"
    )
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[...].set(pos))
    )

    # Set up colours for tracking lines
    rng = np.random.default_rng(0)
    rgbas = rng.random((sim.n_drones, 4))
    rgbas[..., 3] = 1
    swarm_pos = [deque(maxlen=100) for _ in range(sim.n_drones)]
    # Start music if a song is specified
    if music_manager is not None and gui:
        music_manager.play()
        ...

    # MAIN SIMULATION LOOP
    path = Path(__file__).parents[2] / "video/100/raw"
    path.mkdir(parents=True, exist_ok=True)
    # resolution = (240 * 16 // 9, 240)
    resolution = (3840, 2160)
    resolution = (1920, 1080)
    # default_cam_config = {"distance": 7, "elevation": -40, "azimuth": 20}  # 10
    # default_cam_config = {"distance": 20, "elevation": -40, "azimuth": 20}  # 50
    default_cam_config = {"distance": 30, "elevation": -30, "azimuth": 20}  # 100
    save_video = True
    tstart = time.time()
    from tqdm import tqdm

    img_cnt = 0
    for i in tqdm(range(0, int(t * sim.control_freq))):
        current_time = i / sim.control_freq
        des_pos = np.array([[s(current_time) for s in splines[j]] for j in splines])
        des_vel = np.array([[s(current_time) for s in vel_splines[j]] for j in splines])
        controls = np.concatenate((des_pos, des_vel, np.zeros((sim.n_drones, 7))), axis=-1)[
            None, ...
        ]
        # Updates Simulation data
        sim.state_control(controls)
        sim.step(sim.freq // sim.control_freq)

        # Set up tracking lines that show the future drone positions
        if (((i * fps) % sim.control_freq) < fps) and gui:
            for j, dq in enumerate(swarm_pos):
                dq.append(np.asarray(sim.data.states.pos[0, j]))
                draw_line(sim, np.array(dq), rgba=rgbas[j % len(rgbas)], min_size=2, max_size=5)
            # horizon = current_time - np.linspace(0, 2, 20)
            # des_pos = np.array([[s(horizon) for s in splines[i]] for i in splines])
            # for j, traj in enumerate(des_pos):
            #     draw_line(sim, traj.T, rgba=rgbas[j % len(rgbas)])
            if save_video:
                img = sim.render(
                    mode="rgb_array",
                    default_cam_config=default_cam_config,
                    width=resolution[0],
                    height=resolution[1],
                )
                PIL.Image.fromarray(img).save(path / f"img_{img_cnt:05d}.png")
                img_cnt += 1
            else:
                sim.render()
            if (dt := current_time - (time.time() - tstart)) > 0:
                time.sleep(dt)
    sim.close()
