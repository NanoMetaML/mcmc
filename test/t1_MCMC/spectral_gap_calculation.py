import torch
import functools
import mcmc
import transition_matrix
import scipy
import numpy as np

H = torch.randn((6, 6))
temperatures = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
spectral_gaps = []
for temp in temperatures:
    tm = transition_matrix.calc_transition_matrix(n=6, H=H, temperature=temp).squeeze()
    w, v = np.linalg.eig(tm)
    w = np.sort(w)
    spectral_gaps.append(1 - max(w[w != 1]))

import matplotlib.pyplot as plt
# log-log plot

print(spectral_gaps)
plt.figure()
plt.loglog(temperatures, spectral_gaps)
plt.xlabel('Temperature')
plt.ylabel('Spectral gap')
plt.savefig('./test/data/t5_uniform_MCMC_plot.png')
