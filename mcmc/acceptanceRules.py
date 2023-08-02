
def MetropolisHastings(s, s_p, prob):
    """
    Metropolis-Hastings acceptance rule.
    Parameters
        s: current state
        s_p: proposed state
        prob: function that returns the probability of a state

    """


    acceptance_prob = min(1, prob(s_p))
    if torch.rand(1) < acceptance_prob:
        s = s_p

    return s


