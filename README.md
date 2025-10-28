# SwarmGPT

![swarm_gpt_banner](/docs/img/swarm_gpt_banner.png)
[![Format Check](https://github.com/utiasDSL/swarmGPT/actions/workflows/ruff.yaml/badge.svg)](https://github.com/utiasDSL/swarmGPT/actions/workflows/ruff.yaml)
[![website](https://github.com/utiasDSL/swarmGPT/actions/workflows/website.yaml/badge.svg)](https://github.com/utiasDSL/swarmGPT/actions/workflows/website.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

SwarmGPT integrates large language models (LLMs) with safe swarm motion planning, providing an automated and novel approach to deployable drone swarm choreography. Users can automatically generate synchronized drone performances through natural language instructions. Emphasizing safety and creativity, the system combines the creative power of generative models with the effectiveness and safety of model-based planning algorithms. For more information, visit the [project website](https://utiasdsl.github.io/swarm_GPT_dev/) or read our [paper](https://ieeexplore.ieee.org/document/11197931/).

- [Installation](#installation)
- [How to run SwarmGPT](#how-to-run-swarmgpt)
- [Deployment](#deployment)
- [Citing](#citing)

## Installation

[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)

SwarmGPT uses [Pixi](https://pixi.sh) for dependency management and environment setup. Pixi provides a fast, reliable package manager that handles both conda and PyPI dependencies seamlessly.

### Prerequisites

- Linux x64 system (required for ROS Noetic support)
- [Pixi package manager](https://pixi.sh) - see [installation instructions](https://pixi.sh/latest/installation/)

### Setting up SwarmGPT

1. Clone the repository:
   ```bash
   git clone https://github.com/utiasDSL/swarmGPT.git
   cd swarmGPT
   ```

2. Install dependencies and set up the environment:
   ```bash
   pixi install
   ```

3. Activate the environment:
   ```bash
   pixi shell
   ```

The environment includes:
- **Python 3.11** with essential scientific computing packages
- **ROS Noetic Desktop** for robot communication and control
- **Build tools** (cmake, ninja, catkin_tools) for ROS workspace compilation
- **Development tools** (ruff for linting, uv for fast Python package management)
- **Point Cloud Library (PCL)** for 3D processing

### Documentation Environment

To work with documentation, use the docs environment:

```bash
# Serve documentation locally
pixi run -e docs docs-serve

# Build documentation
pixi run -e docs docs-build
```
## How to run SwarmGPT

### Prerequisites

Before running SwarmGPT, ensure you have:

1. **OpenAI API Key**: Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Crazyswarm Configuration**: Configure your drone swarm by editing the `crazyflies.yaml` file in your Crazyswarm installation. SwarmGPT automatically locates this file at:
   ```
   <crazyswarm_path>/launch/crazyflies.yaml
   ```
   
   This file defines:
   - Drone IDs and radio addresses
   - Initial positions for each drone in the swarm
   - Flight area boundaries

### Launching the Interface

1. **Activate the Pixi environment** (if not already active):
   ```bash
   pixi shell
   ```

2. **Launch SwarmGPT**:
   ```bash
   python swarm_gpt/launch.py
   ```
   
   Optional parameters:
   ```bash
   # Use different LLM model
   python swarm_gpt/launch.py --model_id="gpt-3.5-turbo"
   
   # Disable motion primitives (use raw waypoints)
   python swarm_gpt/launch.py --use_motion_primitives=False
      ```

3. **Access the web interface**: The terminal will display a local URL (typically `http://127.0.0.1:7860`). Open this link in your web browser.

### Using the Interface

1. **Select a song** from the available music library
2. **Generate choreography** - SwarmGPT will create a first synchronized drone performance automatically
3. **Preview the results** in the simulation viewer
4. **Refine as needed** by providing additional prompts or modifications
5. **Deploy when satisfied** with the generated choreography

The system will automatically:
- Analyze the selected music for beats, rhythm, and musical features
- Generate safe, collision-free trajectories for your drone swarm
- Ensure all movements stay within the configured flight boundaries
- Synchronize drone movements with the musical timeline

### Ready for Deployment

Once you're happy with your generated choreography, you can proceed to deploy it on your physical drone swarm.

## Deployment

To deploy the generated choreography on your physical drone swarm, crazyswarm has to be running **before** starting the SwarmGPT interface!

1. **Start the Crazyswarm server**:
   ```bash
   roslaunch crazyswarm hover_swarm.launch
   ```
2. **Launch SwarmGPT** as described in the [Launching the Interface](#launching-the-interface) section.
3. **Generate and preview choreography** using the web interface.
4. **Deploy to drones**: Once satisfied with the choreography, click the "Let the Crazyflies dance" button in the web interface to execute the performance on your physical drone swarm.


## Citing
If you find this work useful, compare it with other approaches or use some components, please cite
us as follows:

```bibtex
@article{schuck2025swarmgpt,
  title={SwarmGPT: Combining Large Language Models with Safe Motion Planning for Drone Swarm Choreography},
  author={Schuck, Martin and Dahanaggamaarachchi, Dinushka Orrin and Sprenger, Ben and Vyas, Vedant and Zhou, Siqi and Schoellig, Angela P.},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```