#!/usr/bin/env python3
"""Implement CapsNet architecture.
"""
from torch import Tensor, zeros
from torch.autograd import Variable
from torch.nn import Conv2d, Module, Parameter, ReLU, Sequential
from torch.nn.functional import softmax


class Capsules(Module):
    """Provide base capsule operation.
    """
    def squash(self, s_j):
        """Introduce non-linearity through squashing.

        Args:
            s_j: The input vector.

        Returns:
            The output vector of the same size.
        """
        s_j_norm = s_j.norm(1)

        return (s_j_norm ** 2 / (1 + s_j_norm ** 2)) * (s_j / s_j_norm)


class PrimaryCapsules(Capsules):
    """Perform a convolutional capsule operation.
    """
    def __init__(self, in_channels=256, out_channels=8, out_capsules=32,
                 **kwargs):
        super(PrimaryCapsules, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_capsules = out_capsules

        self.capsules = Conv2d(in_channels=in_channels,
                               out_channels=out_channels * out_capsules,
                               **kwargs)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        x = self.capsules(x)
        x = self.relu(x)
        x = self.squash(x)

        return x.view(-1, self.out_capsules, self.out_channels, *x.size()[2:])


class DigitCapsules(Capsules):
    """Perform a fully-connected capsule operation.
    """
    def __init__(self, in_capsules=32, in_channels=8, out_capsules=10,
                 out_channels=16, num_routing=3):
        super(DigitCapsules, self).__init__()

        self.in_capsules = in_capsules
        self.in_channels = in_channels
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.num_routing = num_routing

        self.register_parameter('W', None)

    def _mul_1(self, u, W):
        """Multiply tensors of different sizes, namely u and W.

        Args:
            u: The output of the previous capsule, of size [batch_size,
            in_capsules, in_channels, width, height].
            W: The weight matrix, of size [in_capsules, in_channels, width,
            height, out_capsules, out_channels].

        Returns:
            The prediction vectors, of size [batch_size, in_capsules,
            (in_channels * width * height), out_capsules, out_channels]
        """
        width, height = W.size()[2:4]

        u = u.view(*u.size(), 1, 1)
        W = W.view(1, *W.size())

        u_hat = u.expand_as(W) * W
        u_hat = u_hat.view(-1,
                           self.in_capsules,
                           (self.in_channels * width * height),
                           self.out_capsules, self.out_channels)

        return u_hat

    def _mul_2(self, c_ij, u_hat):
        """Multiply tensors of different sizes, namely c_ij and u_hat.

        Args:
            c_ij: The coupling coefficients, of size [in_capsules,
            out_capsules].
            u_hat: The prediction vectors, of size [batch_size, in_capsules,
            (in_channels * width * height), out_capsules, out_channels].

        Returns:
            The total input for the capsule, of size [batch_size, out_capsules,
            out_channels]
        """
        c_ij = c_ij.view(1, self.in_capsules, self.out_capsules, 1)
        u_hat = u_hat.sum(2)

        return (c_ij.expand_as(u_hat) * u_hat).sum(dim=1)

    def _dot(self, u_hat, v_j):
        """Measure the agreement between the current output of each capsule in
        the layer above, and the prediction made by each capsules in the layer
        below.

        Args:
            u_hat: The prediction vectors, of size [batch_size, in_capsules,
            (in_channels * width * height), out_capsules, out_channels].
            v_j: The current output of the each capsule in the layer above, of
            size [batch_size, out_capsules, out_channels].

        Returns:
            The agreement vector, of size [in_capsules, out_capsules].
        """
        u_hat = u_hat.sum(dim=4)
        u_hat = u_hat.sum(dim=2)
        u_hat = u_hat.sum(dim=0)

        v_j = v_j.sum(dim=2)
        v_j = v_j.sum(dim=0, keepdim=True)

        return u_hat * v_j

    def forward(self, u):
        use_cuda = u.is_cuda

        width, height = u.size()[3:5]

        if self.W is None:
            W = Tensor(self.in_capsules, self.in_channels, width, height,
                       self.out_capsules, self.out_channels)
            if use_cuda:
                W = W.cuda()
            self.W = Parameter(W)

        u_hat = self._mul_1(u, self.W)

        b_ij = zeros(self.in_capsules, self.out_capsules)
        if use_cuda:
            b_ij = b_ij.cuda()

        for _ in range(self.num_routing):
            c_ij = softmax(Variable(b_ij), dim=0)
            s_j = self._mul_2(c_ij, u_hat)
            v_j = self.squash(s_j)
            b_ij += self._dot(u_hat, v_j).data

        return v_j


class CapsNet(Module):
    """Learn and predict MNIST dataset through a simple CapsNet architecture.
    """
    def __init__(self):
        super(CapsNet, self).__init__()

        self.conv1 = Sequential(Conv2d(in_channels=1,
                                       out_channels=256,
                                       kernel_size=(9, 9),
                                       stride=1),
                                ReLU(inplace=True))
        self.primary_capsules = PrimaryCapsules(kernel_size=(9, 9), stride=2)
        self.digit_capsules = DigitCapsules()

    def forward(self, x):
        x = self.conv1(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)

        return x.norm(dim=2)


class SeparateMarginLoss(Module):
    def __init__(self, margin_positive=0.9, margin_negative=0.1,
                 down_weight=0.5):
        super(SeparateMarginLoss, self).__init__()

        self.margin_positive = margin_positive
        self.margin_negative = margin_negative
        self.down_weight = 0.5

    def forward(self, x, y):
        m_pos = Tensor([self.margin_positive]).view(1, 1).expand_as(x)
        m_neg = Tensor([self.margin_negative]).view(1, 1).expand_as(x)
        L = self.down_weight

        if x.is_cuda:
            m_pos = m_pos.cuda()
            m_neg = m_neg.cuda()

        return ((y * (m_pos - x).clamp(min=0).pow(2))
                + (L * (1 - y) * (x - m_neg).clamp(min=0)).pow(2))
