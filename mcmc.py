import numpy as np
from utils import *


def metropolis_hastings(transition):
    histories = [np.random.randint(0, transition.shape[0])]
    for i in range(N_ITERATIONS - 1):
        current = histories[-1]
        probabilities = transition[current]
        choice = np.random.choice(np.arange(0, transition.shape[0]), p=probabilities)
        histories.append(choice)
    return histories
