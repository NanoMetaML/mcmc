
def sample(sampler, prior, num_samples):
    samples = []
    for i in range(num_samples):
        samples.append(sampler(prior(i)))
    return samples
