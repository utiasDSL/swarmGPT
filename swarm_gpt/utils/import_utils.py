"""Import utility package to deal with any crazyswarm import issues."""

import logging
import sys

from swarm_gpt.utils.utils import get_ros_package_path

logger = logging.getLogger(__name__)


try:
    import pycrazyswarm  # noqa: F401
except ImportError:
    path = get_ros_package_path("crazyswarm", heuristic_search=True)
    pycrazyswarm_path = path / "scripts"
    if str(pycrazyswarm_path) not in sys.path:
        sys.path.insert(0, str(pycrazyswarm_path))

    import pycrazyswarm  # noqa: F401

try:
    from crazyswarm.msg import Position  # noqa: F401
except ImportError:
    # Mock the import of crazyswarm in case we are only running in sim, i.e. without ROS
    logger.warning("Crazyswarm not installed. Mocking import with simple namespace object.")
    Position = None
