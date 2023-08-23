import torch


class MCMCLayer(torch.nn.Module):
    """
    MCMC Layer
        
    Parameters
    sampler: function
        A function that takes in a tensor and returns a tensor
    acceptance_rule: function
        A function that takes in two tensors and returns a tensor
    steps: int
        Number of steps to run the MCMC chain

    Returns
    x: tensor
        The final state of the MCMC chain

    Methods
    forward(x, steps=10)
        Runs the MCMC chain for the specified number of steps
    validate(x, steps=10, **kwargs)
        Runs the MCMC chain for the specified number of steps and returns a 
        batched tensor of all final states and proposed states
    """

    def __init__(self, sampler, acceptanceRule, steps=10):
        super(MCMCLayer, self).__init__()
        self.sampler = sampler
        self.acceptanceRule = acceptanceRule

    def forward(self, x, steps=10, **kwargs):
        """
        Runs the MCMC chain for the specified number of steps and returns
        the final state
        """

        for i in range(steps):
            x_hat = self.sampler(x)
            x = self.acceptanceRule(x, x_hat)

        return x

    def validate(self, x, steps=10, **kwargs):
        """
        Runs the MCMC chain for the specified number of steps and returns 
        a batched list of tensors of all final states and proposed states
        """
        x_list = [x]
        x_hat_list = []

        for i in range(steps):
            x_hat = self.sampler(x)
            x_hat_list.append(x_hat)
            x = self.acceptanceRule(x, x_hat)
            x_list.append(x)

        return x_list, x_hat_list
