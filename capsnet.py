#!/usr/bin/env python3
"""
"""
from torch import Tensor, zeros
from torch.nn import Conv2d, Module, Parameter, ReLU
from torch.nn.functional import softmax


def squash(s):
    """Squash a vector.
    """
    s_norm = s.norm(1)

    return (s_norm ** 2 / (1 + s_norm ** 2)) * (s / s_norm)


class PrimaryCapsule(Module):
    """
    """
    def __init__(self, num_units=9, num_outputs=32, kernel_size=9):
        super(PrimaryCapsule, self).__init__()

        self.capsules = Conv2d(num_units * num_outputs,
                               kernel_size=kernel_size)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        x = self.capsules(x)
        x = self.relu(x)
        x = x.view(-1, 1152, 8, 1)
        x = squash(x)

        return x


class DigitCapsule(Module):
    """
    """
    def __init__(self, num_routing):
        super(PrimaryCapsule, self).__init__()

        self.num_routing = num_routing

        self.W_ij = Parameter(Tensor(1, 1152, 160, 8, 1))

    def forward(self, u):
        u = u.view(-1, 1152, 1, 8, 1)
        u_hat = u * self.W_ij.expand_as(u)

        b_ij = zeros(1, 1152, 10, 1)

        for _ in range(self.num_routing):
            c_ij = softmax(b_ij, dimension=2)
            s_j = (c_ij.expand_as(u_hat) * u_hat).sum(dimension=1)
            v_j = squash(s_j)
            b_ij += u_hat * v_j

        return v_j
