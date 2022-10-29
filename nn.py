import torch

from torch import nn


class NN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))
        self.layers.append(nn.ReLU())
        for i in range(1, len(hidden_size)):
            self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class PolicyNetDiscrete(NN):
    def __init__(self, in_size, hidden_size, out_size) -> None:
        super().__init__(in_size, hidden_size, out_size)
        self.layers.append(nn.Linear(hidden_size[-1], out_size))
        self.layers.append(nn.Softmax(dim=-1))


class PolicyNetContinuos(NN):
    def __init__(self, in_size, hidden_size, out_size) -> None:
        super().__init__(in_size, hidden_size, out_size)
        self.out_size = out_size
        self.mu_out = nn.Linear(hidden_size[-1], 1)
        self.sigma_out = nn.Linear(hidden_size[-1], 1)

    def forward(self, x):
        x = self.layers(x)
        mu = torch.tanh(self.mu_out(x)) * self.out_size
        sigma = nn.functional.softplus(self.sigma_out(x))
        return mu, sigma


class ValueNet(NN):
    def __init__(self, in_size, hidden_size, out_size) -> None:
        super().__init__(in_size, hidden_size, out_size)
        self.layers.append(nn.Linear(hidden_size[-1], out_size))
