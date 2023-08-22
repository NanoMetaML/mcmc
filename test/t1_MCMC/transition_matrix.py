import mcmc
import torch
import numpy as np
from itertools import product
import math as m
import scipy

def calc_transition_matrix(n, H, temperature, proposal_distribution='uniform'):
    """
    Creates a transition matrix where the rows represent the current state of the Markov chain and
    the columns represent the possible states to transition to. All entires are the probabilities
    of transitioning to a given state.

    If H is batched (batch_size, n, n), returns transition matrix of shape (batch_size, 2 ** n, 2 ** n)
    If H is singular, either (1, n, n) or (n, n), returns transition matrix of shape (1, 2 ** n, 2 ** n)
    """
    if proposal_distribution == 'uniform':

        if (len(H.size()) == 2):
            H = H.unsqueeze(0)
        
        previous_states = torch.tensor(list(product([0, 1], repeat=n))).float().unsqueeze(0).repeat(H.size()[0], 1, 1)
        transition_states = previous_states
        transition_matrix = torch.zeros((H.size()[0], 2 ** n, 2 ** n))
        proposal_probability = 1 / (2 ** n)

        for i in range(2 ** n):
            for j in range(2 ** n):
                if (i == j):
                    continue

                previous_state_energy = mcmc.energyFns.quboEnergy(previous_states[:, i, :], H)
                transition_state_energy = mcmc.energyFns.quboEnergy(transition_states[:, j, :], H)

                acceptance_probability = torch.min(torch.ones(H.size()[0]), m.e ** ((previous_state_energy - transition_state_energy) / temperature)).detach()

                transition_matrix[:, i, j] = torch.mul(proposal_probability, acceptance_probability)

        for i in range(2 ** n):
            transition_matrix[:, i, i] = 1 - torch.sum(transition_matrix[:, i, :])


    return transition_matrix