import torch


class MCMC(torch.nn.Module):

    def __init__(self, sampler, update_rule, steps=10):
        super(MCMC, self).__init__()
        self.sampler = sampler
        self.update_rule = update_rule
        self.steps = steps

    def forward(self, x, steps=None):

        if steps is None:
            steps = self.steps
        for i in range(steps):
            x_hat = self.sampler(x, i)
            x = self.update_rule(x, x_hat, i)

        return x
