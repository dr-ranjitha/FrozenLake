# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:14:34 2020

@author: drran
"""
import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

#Create the Q-table
state_space_size = env.observation_space.n #rows
action_space_size = env.action_space.n #columns

q_table = np.zeros((state_space_size, action_space_size))

#Initializing Q-Learning Parameters
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

