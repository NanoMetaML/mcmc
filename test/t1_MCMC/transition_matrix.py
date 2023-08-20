import mcmc
import torch
import numpy as np
from itertools import product
import math as m

def calc_transition_matrix(n, H, temperature, proposal_distribution='uniform'):
    """
    Creates a transition matrix where the rows represent the current state of the Markov chain and
    the columns represent the possible states to transition to. All entires are the probabilities
    of transitioning to a given state.
    """

    if proposal_distribution == 'uniform':
        previous_states = torch.tensor(list(product([0, 1], repeat=n))).float()
        transition_states = torch.tensor(list(product([0, 1], repeat=n))).float()
        proposal_probability = 1 / (2 ** n)
        transition_matrix = torch.zeros((n, n))

        for i in range(n):
            for j in range(n):
                previous_state_energy = mcmc.energyFns.quboEnergy(previous_states[i], H)
                transition_state_energy = mcmc.energyFns.quboEnergy(transition_states[j], H)

                acceptance_probability = min(1, m.e ** ((previous_state_energy - transition_state_energy) / temperature))

                transition_matrix[i, j] = proposal_probability * acceptance_probability

        return transition_matrix