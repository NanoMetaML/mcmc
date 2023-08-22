import mcmc
import functools


def main():
    uniformPrior = functools.partial(mcmc.binaryPriors.uniform, n=10, device = 'cpu')
    for i in range(10):
        print(uniformPrior(i=i))

if __name__ == '__main__':
    main()
