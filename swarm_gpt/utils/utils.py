"""Collection of utility functions for the swarm_gpt package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import rospkg
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def get_ros_package_path(pkg: str, heuristic_search: bool = False) -> Path:
    """Get the path to a ROS package.

    If the package is not found and heuristic_search is enabled, we search for the package manually
    in the user's home directory. Any directory with the pattern *_ws is considered a workspace. We
    then check if the crazyswarm folder is present in the src directory of the workspace.

    Args:
        pkg: The name of the ROS package.
        heuristic_search: Flag to enable search heuristics if ROS cannot find the package.

    Returns:
        The path to the ROS package.
    """
    try:
        return Path(rospkg.RosPack().get_path(pkg))
    except rospkg.common.ResourceNotFound as e:
        if not heuristic_search:
            raise e
    logger.info(f"ROS package {pkg} not found. Searching for the package manually.")
    home = Path.home()
    for path in (d for d in home.glob("*_ws") if d.is_dir()):
        if not (path / f"src/{pkg}").is_dir():
            continue
        pkg_path = path / f"src/{pkg}"
        # Check if the installed package is in the old or new layout. Old layout has nested ros_ws
        # directories, new layout has the proper ROS package structure.
        layout = "old" if (pkg_path / "ros_ws").is_dir() else "new"
        if layout == "old":
            pkg_path = pkg_path / f"ros_ws/src/{pkg}"
        if not pkg_path.is_dir():
            continue
        return pkg_path
    raise rospkg.common.ResourceNotFound(f"ROS package {pkg} not found.")


def draw_line(
    sim: Sim,
    points: NDArray,
    rgba: NDArray | None = None,
    min_size: float = 3.0,
    max_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        sim: The crazyflow simulation.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        min_size: The minimum line size. We linearly interpolate the size from min_size to max_size.
        max_size: The maximum line size.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(min_size, max_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def draw_points(sim: Sim, points: NDArray, rgba: NDArray | None = None, size: float = 3.0):
    """Draw points into the simulation.

    Args:
        sim: The crazyflow simulation.
        points: An array of [N, 3] points.
        rgba: The color of the line.
        size: The size of points.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many points. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    size = np.ones(3) * size
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = np.eye(3).flatten()
    for i in range(len(points)):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size, pos=points[i], mat=mats, rgba=rgba
        )


def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    v = p2 - p1
    vnorm = np.linalg.norm(p2 - p1, axis=-1, keepdims=True)
    p1 = np.where(
        vnorm < 1e-6, p1 + 1e-4, p1
    )  # <add eps to points that are identical to avoid singularity issues
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))
