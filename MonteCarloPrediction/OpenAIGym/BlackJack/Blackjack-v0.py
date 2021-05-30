"""""
Written by Anthony Meyer - 2021
Please do not copy any of my code for school assignments. 
You won't gain anything in life by copying other people's work. 
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
gamma = 0.9999
epsilon = 1.0
alpha = 0.01

# Initialize the environment
env = gym.make('Blackjack-v0')

# Initialize the value of each state to be 0
# Player gets 12-21
V = { }
for playersHand in range(12, 22):
    for dealersHand in range(1, 11):
        for usableAce in range(0, 2):
            V[(playersHand, dealersHand, usableAce)] = 0.0
R = { } # A dict of lists for each possible state visited

# Play 1,000 hands
for episode in range(100000):

    # Puts the agent in a known state
    s = env.reset()

    # Generate an episode of events
    E = []
    done = False
    while done == False:

        # Apply policy (if 20 or 21 stay, hit otherwise)
        a = 0 if s[0] >= 20 else 1
        
        # Take action
        obs, reward, done, __ = env.step(a)
        E.append((s, a, reward))
        
        # Update state
        s = obs
    
    # Loop through episodes of 
    G = 0
    for t in E:
        G = gamma*G + t[2]
        if not t[0] in R:

            R[t[0]] = [G]
        else:
            R[t[0]].append(G)
        V[t[0]] = np.average(R[t[0]])
    
    # Print how episode went
    print(V[(21, 1, 0)])

# Close out the environment
env.close()