"""
File: collect_timing.py
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


def benchmarkPackage():
    """
    Main function for test.py

    """
    logging.info("Benchmarking nanomcmc package")

    b = 1000  # batch size
    n = 1000  # number of variables in polynomial
    s = 1000  # number of steps
    d = 5

    num_terms = [n] * d

    orig_coefficients = polytensor.generators.coeffPUBORandomSampler(
        n=n, num_terms=num_terms, sample_fn=lambda: torch.rand(1)
    )

    poly = polytensor.SparsePolynomial(coefficients=orig_coefficients)

    x = torch.bernoulli(torch.ones(b, n) * 0.5)

    uniformBoltzmann = mcmc.Boltzmann(
        proposer=lambda s: torch.bernoulli(torch.ones_like(s) * 0.5),
        energy_fn=lambda x: poly(x),
        steps=s,
    )

    start = time.time()
    uniformBoltzmann(x)
    end = time.time()

    logging.info(f"Time elapsed: {end - start} seconds")
