# thermoAI

This project is created to provide a general heating system controller with Reinforcement Learning. 

Main goals:
* Is it possible to controll with RL safely --> hold the temperatures in the predefined range
* Is it possible to be more optimal --> reduce cost

## Modules
### Simulator
This is the most important part of the training for the predefined heat-model and data-driven model-based RL as well
### Controller
The collection of well known controlling tools and RL tools.

Controlling tools:
* Classic controlling: 
    * PID with minor modifications for this environment
* Reinforcement learning (future):
    * Deep Deterministic Policy Gradient
    * PPO
    * Soft Actor-Critic
