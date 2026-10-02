<h1 align="center">🔥 Wildfire-Suppressing Multi-Drone Agent</h1>

<p align="center">
  <b>Four firefighting drones that learn to work together against a simulated wildfire, trained with reinforcement learning (PPO).</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Gymnasium-0081A5?style=for-the-badge" alt="Gymnasium"/>
  <img src="https://img.shields.io/badge/PyGame-3F8F3F?style=for-the-badge" alt="PyGame"/>
</p>

<!-- Add a GIF or screenshot of the PyGame simulation here (this is the most important image in the README):
<p align="center"><img src="assets/simulation.gif" width="700" alt="Drones fighting a simulated wildfire"/></p>
-->

## Overview

A custom Gymnasium environment simulates a wildfire spreading across terrain. A team of four drones, three carrying water and one carrying fire retardant, is trained with Proximal Policy Optimization (PPO) to contain and put out the fire. You can watch the trained drones work in a live PyGame window.

## The environment

- **Fire spread:** a cellular automaton over a grid, influenced by moisture, fuel and elevation, plus wind
- **Drones:** three water drones and one retardant drone acting together
- **Resource limits:** every move costs battery, and drones must fly back to the base to refill
- **Visualization:** a PyGame view of the fire, the terrain and the drones

## The agent

- **Algorithm:** Proximal Policy Optimization (PPO), implemented in PyTorch
- **Training:** many training runs of 20 to 200 updates each, with an evaluation script for each experiment
- **Checkpoints:** trained models are saved as `.pth` files

## Quick start

```bash
git clone https://github.com/Mohd-Hanan/Wildfire-Suppressing-Multi-Drone-Agent.git
cd Wildfire-Suppressing-Multi-Drone-Agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python stage_testing.py
```

A PyGame window opens and shows the drones fighting the fire.

## Repository layout

```
├── src/            # environment and agent code
├── configs/        # configuration files
├── tests/          # tests
├── scripts/        # helper scripts
├── stage7_*        # training runs, evaluations, logs and checkpoints
└── stage_testing.py  # runs the trained drones in the PyGame simulation
```

## Why it was interesting to build

- Designing a reward that makes several drones cooperate instead of all chasing the same flames
- Modelling battery and refill limits so that drones have to plan trips
- Keeping training stable across many experiments and comparing runs against each other

## Team

Built as a four-person team project.
