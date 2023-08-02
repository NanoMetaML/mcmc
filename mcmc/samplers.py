import torch
import random
import functools
import enum

from .flips import 

class Basis(enum.Enum):
    spin = 1
    standard = 2

# ---------------------------- SAMPLER FUNCTION BUILDER ---------------------------- #
# The following function is a builder function that takes in a flip function and returns a sampler function
def samplerFromFlipBuilder(
                    x,                              # input tensor
                    flip_fn,                        # flip function
                    basis: Basis = Basis.standard,  # basis of the input tensor
                    grad: bool = False,             # whether to allow gradient 
                                                    # to flow through the flip
                    **kwargs                        # keyword arguments for flip_fn
                    ):
    """
        Builds a sampler function that applies the output of a flip function to a tensor x
    """

    x_flip = flip_fn(x, **kwargs)

    if grad == False:
        x_flip = x_flip.detach()

    if basis == Basis.spin:
        x_noisy = x * (1 - x_flip) + (-1) * x * x_flip

    elif basis == Basis.standard:
        x_noisy = (
            x * (1 - x_flip) + (1 - x) * x_flip
        )
    else:
        raise NotImplementedError("Basis {} not implemented".format(basis))

    return x_noisy


def samplerToFlipBuilder(
                    x,                              # input tensor
                    sampler_fn,                     # sample function
                    **kwargs                        # keyword arguments for flip_fn
                    ):
    """
        Builds a flip function that applies the output of a sampler function to a tensor x
    """

    x_new = sampler_fn(x_s, **kwargs)

    return torch.abs(torch.sign(x_new - x_s)) 


def buildNoisySamplerFn(
                    x,                              # input tensor
                    sampler_fn,                     # sample function
                    detach: bool = False,           # whether to detach the output
                    basis: Basis = Basis.standard,  # basis of the input tensor
                    **kwargs                        # keyword arguments for flip_fn and sampler_fn
                    ):
    """
        Builds a sampler function that adds noise onto x such that x_{t} = f(x_{t-1})
    """
    flip_fn = functools.partial(
        samplerToFlipBuilder,
        sampler_fn=sampler_fn,  # sample function
        **kwargs)               # keyword arguments for flip_fn

    return functools.partial(samplerFromFlipBuilder,
                             detach=detach,
                             basis=basis,
                             flip_fn=flip_fn)


# ---------------------------- SAMPLER FUNCTIONS ---------------------------- #

biasedFlipCategoricalSampler = functools.partial(
    samplerFromFlipBuilder, flip_fn=categoricalFlip
)


unbiasedFlipChooseKSampler = functools.partial(
    samplerFromFlipBuilder, flip_fn=uniformPermFlip
)


def biasedUniformSampler(x, bias=0.5, grad: bool = False, **kwargs):
    """
        Samples a biased bernoulli distribution over the 
        tensor shape.
    """
    return torch.bernoulli(torch.oneslike(x) * bias)


biasedCategoricalSampler.__doc__ = """
        Sampler function that applies a biased categorical flip to a tensor x. 
        Randomly chooses a single bit along the last dimension and flips it
        with probability p_0 if the bit is 0 and p_1 if the bit is 1.
        Parameters:
            x: input tensor
            bias: bias of the bernoulli distribution
            grad: whether to allow gradient to flow through the flip
    """

uniformChooseKSampler.__doc__ = """
        Sampler function that applies a uniform choose k flip to a tensor x.
        Randomly chooses k bits along the last dimension and flips them.
    """
