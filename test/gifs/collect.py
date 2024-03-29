"""
File: collect.py
Author: Blake Wilson 
Email: wilso692@purdue.edu

Description:
    Tests nanomcmc package with various platforms.
"""

import nanomcmc as mcmc
import logging
import time
import polytensor
import torch
import numpy as np
import cv2
import numpy as np


def proposer(s):
    # Chooses a random bit flip
    s_f = torch.distributions.OneHotCategorical(
        probs=torch.ones_like(s) / s.shape[-1]
    ).sample()
    # Flips the bit
    return torch.remainder(s + s_f, 2)


def create_low_res_poly_bitmap(image_path, width, height):
    # Load the high-resolution image
    img = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if img is None:
        raise FileNotFoundError(f"The image at {image_path} could not be loaded.")

    # Resize the image to the defined low-resolution size
    low_res_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    low_res_img = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2GRAY)

    # If a polygonal approximation is desired, additional steps would be needed here
    # For this example, we're only resizing

    # Return the low-resolution image
    return low_res_img


def collect(t=1.0):
    """
    Main function for test.py

    """
    import numpy as np

    w = 12
    h = 16

    img = create_low_res_poly_bitmap("./test/gifs/8Bit_Mario.png", w, h)

    n = w * h

    b = 100_000
    s = 500  # number of steps
    device = "cuda"

    arr = np.zeros((w, h), dtype=np.float32)

    terms = {}
    for i in range(w):
        for j in range(h):
            terms[(i * h + j,)] = img[j, i].item() / 255.0

    # num_terms = [n] * d

    # orig_coefficients = polytensor.generators.coeffPUBORandomSampler(
    #    n=n, num_terms=num_terms, sample_fn=lambda: torch.rand(1, device=device)
    # )

    poly = polytensor.SparsePolynomial(coefficients=terms, device=device)

    x = torch.bernoulli(torch.ones(b, n, device=device) * 0.5)
    y = proposer(x)

    uniformBoltzmann = mcmc.Boltzmann(
        proposer=proposer,
        energy_fn=lambda x: -1 * poly(x),
        steps=s,
        temperature=t,
    )

    y = uniformBoltzmann(x)
    return torch.mean(y, dim=0)


if __name__ == "__main__":
    p = []
    for t in [0.01, 0.1, 1.0, 10, 100, 1000]:
        print(f"Temperature {t}")
        p.append(collect(t))
    torch.save(p, "./test/gifs/sampled_vecs.npy")
    exit(0)
