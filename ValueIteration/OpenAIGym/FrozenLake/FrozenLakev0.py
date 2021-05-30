"""""
Written by Anthony Meyer - 2021
Please do not copy any of my code for school assignments. 
You won't gain anything in life by copying other people's work. 
"""

import gym
import numpy as np

def printPolicy(Pi):
    for r in range(4):
        for c in range(4):
            state = r*4 + c
            if Pi[state] == 0:
                print("< ", end='')
            elif Pi[state] == 1:
                print("V ", end='')
            elif Pi[state] == 2:
                print("> ", end='')
            else:
                print("^ ", end='')
        print("")

def printValue(V):
    for r in range(4):
        for c in range(4):
            print("%.2f " % V[r*4+c], end="")
        print("")

# Hyperparameters
gamma = 0.9999
convergenceCriteria = 0.00001

# Initialize the environment
env = gym.make('FrozenLake-v0', is_slippery=True)

# Puts the agent in a known state
observation = env.reset()

# Initialize the value of each state to be 0
V = np.zeros(env.nS)
Pi = np.zeros(env.nS)

# For keeping track of how many passes through the whole state space we needed
passes = 0

# For keeping track of how many low level math operations we needed
updates = 0

# We will run algorithm until we meet the convergence criteria
while True:

    # Iterate number of passes taken
    passes += 1

    # We need to know when to stop the algorithm, we will use this
    delta = 0.0

    # Iterate through all possible states
    for state in range(env.nS):

        # We need to store our current V[state] so we can apply gradient at end of iteration
        v = V[state]

        # We have to search for the action with the max expected value in our given state
        maxExpectedValue = float('-inf')

        # For each action we can take in a given state
        for a in range(env.nA):

            # This is the expected value for taking action 'a' while in state 'state'
            expectedValue = 0.0
            
            # For each possible next state given our action
            for ns in env.P[state][a]:

                # P(s, a, s') = ns[0]
                P = ns[0]

                # S' = ns[1]
                Sp = ns[1]

                # R(s') = ns[2]
                R = ns[2]

                # Building our expected value
                expectedValue += P*(R + gamma*V[Sp])

                # Update number of low level math operations needed
                updates += 1

            # Check to see if we have a new best action in our current state
            if expectedValue > maxExpectedValue:

                # Update our "best action" policy for this state
                Pi[state] = a

                # Update max action value
                maxExpectedValue = expectedValue
        
        # Apply max absolute value to V[state]
        V[state] = maxExpectedValue
        
        # Calculate gradient
        delta = max(delta, np.abs(v - V[state]))

    # See if we have converged
    if delta < convergenceCriteria:
        break

# Print out the findings of our algorithm
print("The algorithm terminates in %d passes and %d math operations" % (passes, updates))
printPolicy(Pi)
printValue(V)

env.close()