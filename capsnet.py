#!/usr/bin/env python3
"""Implement and train CapsNet architecture for MNIST dataset.
"""
import os
import pdb
import sys

import torch
from torch import LongTensor, Tensor
from torch.autograd import Variable
from torch.nn import (Conv2d, Linear, Module, MSELoss, Parameter, ReLU,
                      Sequential, Sigmoid)
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from utils import get_loaders, get_logger, get_parser, Trainer


def is_nan(x):
    x = x.sum().data[0]
    return True if x != x else False


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
        s_j_norm = s_j.view(s_j.size()[0], -1).norm(p=1, dim=1)
        s_j_norm = s_j_norm.view(-1, *((1,) * (s_j.dim() - 1)))
        s_j_norm_sq = s_j_norm ** 2

        return ((s_j_norm_sq / (1 + s_j_norm_sq)) * (s_j / (s_j_norm + 1e-7)))


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
        u = u.expand(*u.size()[:-2], *W.size()[-2:])
        W = W.view(1, *W.size())
        W = W.expand_as(u)

        u_hat = u * W
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
                       self.out_capsules, self.out_channels).normal_(std=0.1)
            if use_cuda:
                W = W.cuda()
            self.W = Parameter(W)

        u_hat = self._mul_1(u, self.W)

        b_ij = torch.zeros(self.in_capsules, self.out_capsules)
        if use_cuda:
            b_ij = b_ij.cuda()

        for _ in range(self.num_routing):
            c_ij = softmax(Variable(b_ij), dim=1)
            s_j = self._mul_2(c_ij, u_hat)
            v_j = self.squash(s_j)
            b_ij += self._dot(u_hat, v_j).data
            if is_nan(c_ij) or is_nan(s_j) or is_nan(v_j):
                pdb.set_trace()

        return v_j


class CapsNet(Module):
    """Learn and predict MNIST dataset through a simple CapsNet architecture.

    On forward pass, this model outputs a tuple of the input data and the
    the output of digit capsules.
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
        y = self.conv1(x)
        y = self.primary_capsules(y)
        y = self.digit_capsules(y)

        return x, y


class SeparateMarginLoss(Module):
    """Calculate margin loss separately for existence and absence.

    Args:
        margin_positive (float): The positive margin.
        margin_negative (float): The negative margin.
        down_weight (float): The negative margin weight.
    """
    def __init__(self, margin_positive=0.9, margin_negative=0.1,
                 down_weight=0.5):
        super(SeparateMarginLoss, self).__init__()

        self.margin_positive = margin_positive
        self.margin_negative = margin_negative
        self.down_weight = down_weight

    def _encode(self, y, num_classes):
        t = torch.zeros(len(y), num_classes).type(LongTensor)

        batch_i = torch.arange(0, len(y)).type(LongTensor)
        class_i = y.data.type(LongTensor)

        t[batch_i, class_i] = 1

        return t

    def forward(self, x, y):
        tensor_type = type(x.data)

        m_pos = tensor_type([self.margin_positive]).view(1, 1).expand_as(x)
        m_neg = tensor_type([self.margin_negative]).view(1, 1).expand_as(x)
        L = self.down_weight

        t = self._encode(y, x.size()[1])
        t = Variable(t.type(tensor_type))

        loss = ((t * (Variable(m_pos) - x).clamp(min=0).pow(2))
                + (L * (1 - t) * (x - Variable(m_neg)).clamp(min=0).pow(2)))

        return loss.sum(dim=1)


class ReconstructionLoss(Module):
    """Calculate the MSE between reconstruction and the input.

    Args:
        in_capsules (int): The number of input capsules.
        in_channels (int): The number of input channels.
    """
    def __init__(self, in_capsules=10, in_channels=16,
                 out_channels=(1 * 28 * 28)):
        super(ReconstructionLoss, self).__init__()

        self.in_capsules = in_capsules
        self.in_channels = in_channels

        self.fc = Sequential(Linear(in_capsules * in_channels, 512),
                             ReLU(inplace=True),
                             Linear(512, 1024),
                             ReLU(inplace=True),
                             Linear(1024, out_channels),
                             Sigmoid())
        self.mse = MSELoss()

    def forward(self, x, y):
        x = x.view(-1, self.in_capsules * self.in_channels)
        x = self.fc(x)

        return self.mse(x, y)


class CapsNetLoss(Module):
    """Calculate the separate margin loss and reconstruction loss.

    Args:
        recon_weight (float): The reconstruction loss weight.
    """
    def __init__(self, recon_weight=0.0005):
        super(CapsNetLoss, self).__init__()

        self.recon_weight = recon_weight

        self.separate_margin = SeparateMarginLoss()
        self.recon = ReconstructionLoss()

    def forward(self, x, y):
        x, output = x

        loss = self.separate_margin(output.norm(dim=2), y)

        x = Variable(x.data, requires_grad=False)
        y = Variable(y.data.type(type(output.data)))
        output = output * y.view(*y.size(), 1, 1).expand_as(output)
        loss += self.recon_weight * self.recon(output, x)

        return loss.sum()


def main():
    """Train and validate CapsNet architecture on MNIST dataset.
    """
    # Prepare logger and argument parser.

    logger = get_logger()
    parser = get_parser(__doc__)
    args = parser.parse_args(sys.argv[1:])
    logger.info(' '.join(sys.argv))

    # Prepare checkpoints and datasets directory.

    for directory in ('checkpoints', 'checkpoints/capsnet',
                      'datasets', 'datasets/mnist'):
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

    # Prepare data loaders.

    logger.info('Prepare data')

    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    train_dataset = MNIST('datasets/mnist',
                          train=True,
                          transform=transform,
                          download=True)
    test_dataset = MNIST('datasets/mnist',
                         train=True,
                         transform=transform,
                         download=True)

    train_loader, val_loader, test_loader = get_loaders(train_dataset,
                                                        test_dataset,
                                                        args)

    # Prepare model.

    logger.info('Prepare model')

    capsnet = CapsNet()
    criterion = CapsNetLoss()
    optimizer = Adam(capsnet.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.96)
    trainer = Trainer(
        capsnet, criterion, optimizer, scheduler,
        prediction=lambda o: torch.max(o[1].cpu().data.norm(p=2, dim=2), 1)[1])
    if args.checkpoint_file:
        trainer.load(args.checkpoint_file)

    # Train model.

    logger.info('Begin training')
    epoch_log = ('[epoch {epoch}/{total_epoch}] '
                 'mean training loss: {loss:.3f}, '
                 'validation accuracy: {accuracy:.3f}')

    for epoch in range(args.num_epochs):
        train_loss = trainer.train_epoch(train_loader, args.no_progress_bar)
        val_accuracy = trainer.infer_epoch(val_loader, args.no_progress_bar)

        trainer.save('checkpoints/capsnet/epoch_{epoch}'.format(epoch=epoch+1))

        logger.info(epoch_log.format(epoch=epoch+1,
                                     total_epoch=args.num_epochs,
                                     loss=train_loss,
                                     accuracy=val_accuracy))

    # Test and save final model.

    logger.info('Begin test')
    test_accuracy = trainer.infer_epoch(test_loader, args.no_progress_bar)

    logger.info('Final test accuracy: {accuracy:.3f}'.format(
        accuracy=test_accuracy))

    num = 1
    result_format = 'checkpoints/capsnet/result_{num}.pth'
    while os.path.isfile(result_format.format(num=num)):
        num += 1
    trainer.save(result_format.format(num=num))
    logger.info('Final model saved as: checkpoints/result_{num}.pth'.format(
        num=num))


if __name__ == '__main__':
    main()
