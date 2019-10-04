from __future__ import absolute_import
import sys
sys.path.append(sys.path[0] +"/..")

import numpy as np
import gym
from controller.RL.DDPG import DDPG
from controller.RL.PPO import PPO


class simulatorWrapper():
    def __init__(self,env):
        self.env = env
        self.env.seed(42)
        self.env.reset()
    
    def step(self, action):
        self.counter += 1
        self.state, reward, done, _ = self.env.step([action])
        self.env.render()
        done = done or self.counter>200
        return done, reward, self.state

    def reset(self):
        self.state = self.env.reset()
        self.counter=0

    def get_concated_features(self):
        return self.state

def full_training_DDPG():
    sim = simulatorWrapper(gym.make('Pendulum-v0'))
    ddpg = DDPG(3,-2.0,2.0)
    ddpg.train(sim,init_step=0, episode=1000,batch_size=128)

def full_training_PPO():
    sim = simulatorWrapper(gym.make('Pendulum-v0'))
    ppo = PPO(3, -2.0, 2.0)
    ppo.train(sim, init_step=0, episode=1000, batch_size=128)


if __name__== "__main__":
  print('Train with DDPG')
  #full_training_DDPG()
  print('Train with PPO')
  full_training_PPO()

