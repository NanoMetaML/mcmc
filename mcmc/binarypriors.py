import torch


def uniform(n, device, bias=0.5, **kwargs):
    """Uniform prior on the interval [0,1]."""
    return torch.bernoulli(torch.ones(n, device=device) * bias)
