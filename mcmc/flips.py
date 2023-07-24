import torch
import random


def categoricalFlip(x, p_0=0.1, p_1=0.1):
    cat_probs = torch.ones_like(x) * 1 / x.shape[-1]
    neighbors = torch.distributions.onehotcategoricalstraightthrough(
        cat_probs
    ).sample(x.shape[:-2])

    x_0_mask = (x == 0).float() * neighbors
    x_0_flip = x_0_mask * \
        torch.bernoulli(p_0 * torch.ones_like(x_0_mask))

    x_1_mask = (x == 1).float() * neighbors
    x_1_flip = x_1_mask * \
        torch.bernoulli(p_1 * torch.ones_like(x_1_mask))

    x_flip = x_0_flip + x_1_flip

    # sanity check to make sure that the gradient is not flowing through the x_flip
    x_noisy = (
        x.detach() * (1 - x_flip.detach()) + (1 - x.detach()) * x_flip.detach()
    )
    return x_noisy


def multinomialchannel(x, k):

    indices = torch.tensor(random.sample(x.shape[2], k))
    sampled_values = values[indices]

    cat_probs = torch.ones_like(x) * 1 / x.shape[-1]
    neighbors = torch.distributions.onehotcategoricalstraightthrough(
        cat_probs
    ).sample(x.shape[:-2])

    x_0_mask = (x == 0).float() * neighbors
    x_0_flip = x_0_mask * \
        torch.bernoulli(p_0 * torch.ones_like(x_0_mask))

    x_1_mask = (x == 1).float() * neighbors
    x_1_flip = x_1_mask * \
        torch.bernoulli(p_1 * torch.ones_like(x_1_mask))

    x_flip = x_0_flip + x_1_flip

    # sanity check to make sure that the gradient is not flowing through the x_flip
    x_noisy = (
        x.detach() * (1 - x_flip.detach()) + (1 - x.detach()) * x_flip.detach()
    )
    return x_noisy

if __name__ == "__main__":
    k = 2
    x = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float()
    indices = torch.tensor(random.sample(x.shape[2], k))
    print(indices)
