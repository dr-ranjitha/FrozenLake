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
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        # Set new state
        state = new_state        
        # Add new reward   
        rewards_current_episode += reward
        
        if done == True: 
            break

    # Exploration rate decay  
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)
    
# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

********Average reward per thousand episodes********

1000 :  0.16800000000000012
2000 :  0.32800000000000024
3000 :  0.46900000000000036
4000 :  0.5350000000000004
5000 :  0.6580000000000005
6000 :  0.6910000000000005
7000 :  0.6470000000000005
8000 :  0.6550000000000005
9000 :  0.6980000000000005
10000 :  0.7000000000000005

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)
