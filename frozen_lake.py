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
exploration_decay_rate = 0.001

#Hold all rewards from all episodes
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode): 
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) #exploit 
        else:
            action = env.action_space.sample() #explore
        # Take new action
        new_state, reward, done, info = env.step(action)
        # Update Q-table
        # Set new state
        # Add new reward        

    # Exploration rate decay   
    # Add current episode reward to total rewards list

