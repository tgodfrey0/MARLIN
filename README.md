# MARLIN

*Toby Godfrey, William Hunt, Mohammad D. Soorati*

This repository accompanies our paper [**"MARLIN: Multi-Agent Reinforcement Learning Guided by Language-Based, Inter-Robot Negotiation"**]()

*Currently under review for ICRA 2025*

## Usage

To train a model using MARLIN based on MAPPO (Multi-Agent PPO with a centralised critic), simply run

```bash
python src/marl/llm_marl_critic.py --llm <LLM> --timestep-total <TOTAL_NUM_TIMESTEPS> --scenario <SCENARIO_NAME>
```

For example

```bash
 python src/marl/llm_marl_critic.py --llm meta-llama/Meta-Llama-3.1-8B-Instruct --timestep-total 500000 --scenario Maze_Like_Corridor
```

### LLM-Options

- `gpt-X*`
- `gemini-X*`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`

### Scenarios

We created 5 scenarios to test the system under different environments.

<table>
  <tr>
    <td><img src="imgs/asym_env.svg" height="250"><b>Asymmetrical Two Slot Scenario</b></td>
    <td><img src="imgs/sym_env.svg" height="250"><b>Symmetrical Two Slot Scenario</b></td>
    <td><img src="imgs/single_env.svg" height="250"><b>Single Slot Scenario</b></td>
    <td><img src="imgs/two_env.svg" height="250"><b>Two Path Scenario</b></td>
    <td><img src="imgs/maze_env.svg" height="250"><b>Maze-Like Scenario</b></td>
  </tr>
</table>

## Hardware Deployment

Once a model has been trained it can be deployed to real hardware. We tested this on two TurtleBot3 robots running ROS2 Humble Hawksbill.

```bash
ros2 launch cmd_listener run_listener.launch.py
```

This will test all three methods (MARL, LLM, MARLIN) for several checkpoints. The model folder and output folder must be set in `marlin_ros2_ws/src/cmd_listener/cmd_listener/listener.py`.
