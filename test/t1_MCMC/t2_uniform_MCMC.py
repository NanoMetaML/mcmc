import mcmc
import itertools
import functools
import torch
import matplotlib.pyplot as plt


def pick_items_from_dict_lists(dict_of_lists):
    # Get the keys and lists from the dictionary
    keys = list(dict_of_lists.keys())
    lists = list(dict_of_lists.values())

    # Generate the Cartesian product of all lists
    combinations = list(itertools.product(*lists))

    # Create a list of key-value pairs for each combination
    result = [{keys[i]: item[i] for i in range(len(keys))} for item in combinations]
    return result

def pick_items_from_lists(list_of_lists):
    # Generate the Cartesian product of all lists
    combinations = list(itertools.product(*list_of_lists))
    return combinations

def acceptancesampler(s, s_p, energy_fn):

    b = torch.exp(energy_fn(s) - energy_fn(s_p))
    A = min(1, b)
    if A >= torch.rand(1):
        s = s_p
    return s

def quboPrior(n):
    """
    Given size n, generates an (n, n) matrix with random values. Function returns both
    the random matrix H and the qubo energy function.
    """
    H = torch.rand(n, n)
    energy_fn = functools.partial(mcmc.energyFns.quboEnergy, H = H)
    return H, energy_fn

# Simple MCMC sampler
def main(n, init_s_prior, quboPrior, getProposal, sampleAccept, temps):

    data = {}

    for run_idx in range(10):

        H, energy_fn = quboPrior(n)

        data[run_idx] = {
            'spectral_gap': mcmc.energyFns.spectralGap(energy_fn, n),
            'probleminstance': [H, energy_fn],
        }

        for temperature in temps:
            print(f"QUBO: {run_idx} Temperature: {temperature}" )

            s = init_s_prior()                          

            data[run_idx][temperature] = {
                's': [s],
                'energy': [energy_fn(s, H=H)]
            }

            energy_fn =  functools.partial(mcmc.energyFns.quboEnergy, H = H / temperature)
            for sample_idx in range(100):
                s_p = getProposal(s=s)
                s = sampleAccept(s, s_p, energy_fn)
                data[run_idx][temperature]['s'].append(s)
                data[run_idx][temperature]['energy'].append(energy_fn(s, H=H))
    # Save
    torch.save(data,'test/data/t2_uniform_MCMC.pt')
    visualize_data('test/data/t2_uniform_MCMC.pt')

def visualize_data(filename):
    data = torch.load(filename)
    #Creating energy over time graph for each temperature
    run_idx = 0
    for temperature in temps:
        time_values = []
        energies = []
        for idx, energy in enumerate(data[run_idx][temperature]['energy']):
            time_values.append(idx)
            energies.append(energy)
        plt.plot(time_values, energies, label=f'Temperature {temperature}')
    
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(f'Energy versus Time for Different Temperatures (run_idx = {run_idx})')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 6
    temps = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    uniformPrior = functools.partial(mcmc.binaryPriors.uniform, n=n, device = 'cpu')
    main(n, uniformPrior, quboPrior, getProposal=uniformPrior, sampleAccept=acceptancesampler, temps=temps)