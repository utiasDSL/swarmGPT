---
title: Axswarm
description: Trajectory planner to aid in choreography of drones swarms to music.
---

# Axswarm - Trajectory Planner
[Axswarm :octicons-mark-github-24:](https://github.com/utiasDSL/axswarm) is a high-speed, research-grade trajectory planner for drone swarms, now fully reimplemented in Python using JAX for automatic differentiation and GPU/TPU acceleration. The package eliminates the previous C++ core, making installation and usage dramatically simpler—no compilation, no CMake, just Python dependencies. The new API is streamlined and functional, with all core logic and data structures exposed in Python. The codebase is easy to read, extend, and integrate with modern simulation environments.

## Key Features
- Pure [JAX](https://github.com/jax-ml/jax) Implementation: No C++ or pybind11 required. All computation is vectorized and JIT-compiled with JAX.
- Simple, Functional API: Core entry points are just a few functions and data classes.
- Swarm Trajectory Optimization: Efficient, scalable, and suitable for real-time or batch planning.
- Flexible Settings and Data Structures: All configuration and data are Python dataclasses, easy to serialize and modify.
- Easy Installation: No build step—just install Python dependencies.

## Installation
To install:
```bash
git clone https://github.com/utiasDSL/axswarm.git
cd axswarm
pip install -e .
```
(or you can add it to your project and install dependencies from pyproject.toml)

## Usage
See `examples/simulate.py` for a full simulation loop. Here's a minimal example:
```bash
import numpy as np
from axswarm import SolverData, SolverSettings, solve

# Prepare waypoints (shape: dict of arrays, e.g. {"pos": [n_drones, n_points, 3], ...})
waypoints = {
    "time": ...,
    "pos": ...,
    "vel": ...,
    "acc": ...,
}

# Prepare settings (see axswarm/settings.py for all options)
settings = SolverSettings(
    max_iters=20,
    rho_init=1.0,
    rho_max=10.0,
    # ... other settings ...
    pos_min=np.array([-2, -2, 0]),
    pos_max=np.array([2, 2, 2]),
    collision_envelope=np.array([0.3, 0.3, 0.3]),
    # etc.
)

# Initialize solver data
solver_data = SolverData.init(
    waypoints=waypoints,
    K=settings.K,
    N=settings.N,
    A=..., B=..., A_prime=..., B_prime=...,
    freq=settings.freq,
    smoothness_weight=settings.smoothness_weight,
    input_smoothness_weight=settings.input_smoothness_weight,
    input_continuity_weight=settings.input_continuity_weight,
)

# Run the solver for one step
states = ...  # current [n_drones, 6] state (pos, vel)
t = ...       # current time
success, iters, solver_data = solve(states, t, solver_data, settings)
```

## Information about the Package

### Package Structure
- `data.py`: Defines all main data structures, especially `SolverData` (holds all state, waypoints, matrices, etc.).
- `settings.py`: Contains the `SolverSettings` dataclass for all solver and constraint parameters.
- `solve.py`: Implements the main solve function and all optimization logic.
- `constraint.py`, `spline.py`: Helper modules for constraints and trajectory representation.
- `__init__.py`: Exposes the main API: `SolverData`, `SolverSettings`, `solve`.

### API Overview
- `SolverData`: Holds all mutable state for the solver, including waypoints, cost matrices, and current trajectories. Created via SolverData.init(...).
- `SolverSettings`: All solver and constraint parameters, as a dataclass.
- `solve(states, t, data, settings)`: The main functional entry point. Advances the swarm trajectories by one step, returning success flags, iteration counts and updated data.
- Functional, stateless design: All state is explicit; no global variables or hidden state.

### Example Directory
- `examples/simulate.py`: End-to-end simulation of a drone swarm following a spiral formation, including visualization and integration with a simulator
- `examples/utils.py`: Helper functions for visualization.
