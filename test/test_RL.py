from __future__ import absolute_import
import sys
sys.path.append(sys.path[0] +"/..")

import numpy as np
import gym
from controller.RL.DDPG import DDPG
from controller.RL.PPO import PPO
from controller.RL.SAC import SAC
from controller.RL.iLQR import iLQR
from continous_cartpole import ContinuousCartPoleEnv


class simulatorWrapper():
    def __init__(self,env, render=True):
        self.env = env
        self.env.seed(42)
        self.env.reset()
        self.render = render
    
    def step(self, action):
        self.counter += 1
        self.state, reward, done, _ = self.env.step([action])
        if self.render:
            self.env.render()
        done = done or self.counter>=200
        return done, reward, self.state

    def reset(self):
        self.state = self.env.reset()
        self.counter=0

    def get_concated_features(self):
        return self.state

def full_training_DDPG():
    sim = simulatorWrapper(gym.make('Pendulum-v0'),True)
    ddpg = DDPG(3,-2.0,2.0)
    ddpg.train(sim,init_step=0, episode=500,batch_size=128)

def full_training_PPO():
    sim = simulatorWrapper(ContinuousCartPoleEnv(), False)
    ppo = PPO(4, -1.0, 1.0)
    ppo.train(sim, init_step=0, episode=40)
    rewards =[]
    for t in range(100):
        sim.reset()
        done=False
        cum_reward =0
        while not(done):
            act = ppo.control(sim.get_concated_features())
            done, reward, _ = sim.step(act)
            cum_reward += reward
            if t<5:
                sim.env.render()
        rewards.append(cum_reward)
    rewards = np.array(rewards)
    print("Continous cartpole")
    print("Mean of reward(maximum 200):", rewards.mean())
    print("Std of reward:",rewards.std())
    print("Solved from >=195")

def full_training_SAC():
    sim = simulatorWrapper(gym.make('Pendulum-v0'), True)
    sac = SAC(3,-2.0,2.0)
    sac.train(sim,init_step=0, episode=80,batch_size=128)

def full_training_iLQR():
    sim = simulatorWrapper(ContinuousCartPoleEnv(), False)
    model_based = iLQR(4, -1.0, 1.0)
    model_based.train(sim)


if __name__== "__main__":
  print('Train with DDPG')
  #full_training_DDPG()
  print('Train with PPO')
  #full_training_PPO()
  print('Train with SAC')
  full_training_SAC()
  print('Train with iLQR with learned model')
  #full_training_iLQR()

