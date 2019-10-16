# thermoAI

This project is created to provide a general heating system controller with Reinforcement Learning. 

Main goals:
* Is it possible to controll with RL safely --> hold the temperatures in the predefined range
* Is it possible to be more optimal --> reduce cost
* Learn a bit about the continous control

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
Even if my the domain knowledge would be outstanding, it is a really hard problem to create a real-world RL application.

### Simulator
Problems:
* What should be the state features:
    * Previous inside and outside temperatures
    * Expected inside and outside temperature
    * Historical heating power
    * Expected energy prices in the future
* What is the right reward function:
    * Inside temperature penalty:
        * Big penalty for being outside of the desired inside temperature interval
        * MSE from the desired intervall if it is outside (be close if it is not possible being inside)
    * Cost of the used energy
* Using realistic data
    * Weather: I used Basel hourly temperature from 2016-2017 winter
    * Energy cost: Hungarian daytime and night time pricing is used

As this is the most important part of the RL pipeline, it is very important to be bug-free, so sufficient unit test is needed.

### PID Controller
The simulator is linear, so PID should solve the controlling nearly optimal. However it doesn't use the information about the energy price --> it may suboptimal
Problems
* The modelled system is linear and has no delay in the temperature response, which suggests the integrator part and derivator part is useless. Only P must be set.
* The current inside temperature is not enough information, so I combined it with a the outside information
* Knowing the current required temperature is not enough, so a few step forward seeing is required

Otherwise this is not too complicated as the heating system model is linear 

![alt text](img/PID_heat.png "PID heating charasteristics")
This figure shows PID controlls the temperature almost perfectly. The orange part shows the required inside temperature interval. 

### Imitation learning

The goal of the imitation learning was to mimic the PID controlling. It sounds like an easy supervised ML problem... well it isn't that easy...
The NNs are designed to be smooth, but the PID controlling contains spikes, because the target temperature contains step function.

I tried different loss functions:
* Mean Squared Error (MSE)
![alt text](img/mse_pretrain.png "MSE graph")
* Mean Average Error (MAE)
![alt text](img/mae_pretrain.png "MSE graph")
* Adversarial training: The idea is using a nearual network to predict whether state+action pair comes from the PID controller or the imitation learning policy. The policy is trained according to the GAN rule.
![alt text](img/adversarial_pretrain.png "Adversarial graph")
Another interesting aspect of this method is inverse RL. Inverse RL scenario converts a policy and simulator to a reward function. It can be helpful for designing a reward function.
During inverse RL Neural network provide the reward based on action and state. In inverse RL the "discriminator" provides reward function and instead of imitation learning model-free RL happens. This results maximal reward difference for the curret policy and any other policy. 
However the adversarial method performs poor.

According to the results MAE is chosed for being the initial point for model-free RL. P

For Q function based method, it is essential to learn the critic as well.
Another interesting problem about pretraining Actor-Critic architecture is training Critic with "optimal" policy's values causes discrapency between the value function for learned policy and teacher policy.
This problem can be eliminated with learning the policy first and use the learned policy's Q-values for the critic target.
My intuiton: MAE performs thebest as the system is fully linear. 

###Model-Free reinforcement learning
In my experience the heating problem is way too complicated for model free-RL. That is why not converging not necceseraly means that the implementation wrong.
I used OpenAI gym inverted pendulum and continous cartpole task to check convergence. The algorithms works for the given task. This can be unit test of the  implementation.

Without any further trick the model tends to predict one of the corner case of the valid interval.
Due to the previous reasons I decided to use pre-tarining for the models. The baseline is the PID and imitation learning is used as pretrained-model can see in the previous section.

Implemented methods:
* [DDPG](https://arxiv.org/pdf/1509.02971.pdf) 
* [SAC](https://arxiv.org/pdf/1801.01290.pdf)

* [PPO](https://arxiv.org/pdf/1707.06347.pdf) State-of-the-art model-free RL method handles the exporation as well and converge really fast on inverted pendulum problem.

###Model-based reinforcement learning
iLQR method is really slow. The main advantage is able to converge faster than SAC, if not counting the model-learning steps, which are passive steps.
Interesting discovery: TF2.0 can calculate the hessian matrix (d cost/dd input), which is required, but the network must contain non-ReLu activation as well, because the hessian of ReLu network will be yero matrix.




