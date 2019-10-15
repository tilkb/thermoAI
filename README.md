# thermoAI

This project is created to provide a general heating system controller with Reinforcement Learning. 

Main goals:
* Is it possible to controll with RL safely --> hold the temperatures in the predefined range
* Is it possible to be more optimal --> reduce cost

## Modules
### Simulator
This is the most important part of the training for the predefined heat-model and data-driven model-based RL as well. The model is way too simple compared to a normal simulator. The reason behind this is that simulator must be fast. 
### Controller
The collection of well known controlling tools and RL tools.

Controlling tools:
* Classic controlling: 
    * PID with minor modifications for this environment
* Reinforcement learning:
    * Deep Deterministic Policy Gradient (model-free)
    * Proximal Policy Optimization (model-free)
    * Soft Actor-Critic (model-free)
    * itartive Linear Quadratic Regulator (model-based)

## Use
Install dependencies:

```pip install -r requirements.txt```

Train the methods and save the policy:

```python train.py```

Evalute the policies and provide a graph about the performance:

```python evaluate.py```

## Interesting lessons I learned from experimenting

### Simulator

### PID Controller

### Imitation learning
The goal of the imitaion learning was to mimic the PID controlling. It sounds like an easy supervised ML problem... well it isn't that easy...
The NNs are designed to be smooth and the controlling contains spikes.

###Model-Free reinforcement learning
DDPG
PPO
S


