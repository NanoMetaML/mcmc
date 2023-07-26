
import mcmc
import functools
import torch

def main():
    data = {}
    n = 6
    uniformPrior = functools.partial(mcmc.binarypriors.uniform, n=n, device = 'cpu')
    for run_idx in range(10):
        data[run_idx] = {}
        H = torch.rand(n, n)
        energy_fn = functools.partial(mcmc.energy_fn.quboEnergy, H = H)
        data[run_idx]['spectral_gap'] = mcmc.energy_fn.spectralGap(energy_fn, n)
        data[run_idx]['H'] = H
        for temperature in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            print('run_idx: {}, temperature: {}'.format(run_idx, temperature))
            energy_fn =  functools.partial(mcmc.energy_fn.quboEnergy, H = H / temperature)
            data[run_idx][temperature] = {}
            # Initialize mcmc state
            s = uniformPrior()
            data[run_idx][temperature]['s'] = [s]
            for sample_idx in range(10000):
                s_p = uniformPrior(i = sample_idx)
                b = torch.exp(energy_fn(s_p) - energy_fn(s))
                acceptance_prob = min(1, b)
                if torch.rand(1) < acceptance_prob:
                    s = s_p
                data[run_idx][temperature]['s'].append(s)
    # Save
    torch.save(data, 'test/data/t2_uniform_MCMC.pt')

if __name__ == '__main__':
    main()
