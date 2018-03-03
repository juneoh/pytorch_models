#!/usr/bin/env python3
"""Implement and train CapsNet architecture for MNIST dataset.
"""
import os
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

# DocTest parameters.
BATCH_SIZE = 3
IN_CAPSULES = 32
IN_CHANNELS = 8
OUT_CAPSULES = 10
OUT_CHANNELS = 16
NUM_ROUTING = 3
HEIGHT = 20
WIDTH = 20


class DynamicRouting(Module):
    """Provide dynamic routing operation.

    Args:
        in_capsules (int): The number of capsules of the input.
        in_channels (int): The number of channels of the input.
        out_capsules (int): The number of capsules of the output.
        out_channels (int): The number of channels of the output.
        num_routing (int): The number of routing logit update iterations.
    """
    def __init__(self, in_capsules, in_channels, out_capsules, out_channels,
                 num_routing):
        super(DynamicRouting, self).__init__()

        self.in_capsules = in_capsules
        self.in_channels = in_channels
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.num_routing = num_routing

        self.register_parameter('bias', None)

    def _initialize_bias(self, u_hat):
        bias = type(u_hat.data)(self.out_capsules, self.out_channels, 1, 1)
        bias.fill_(0.1)
        self.bias = Parameter(bias)

    def squash(self, s_j):
        """Introduce non-linearity through squashing.

        Args:
            s_j (torch.Tensor): The total output of the layer, of any size.

        Returns:
            (torch.Tensor): The squashed output vector, of the same size.
        """
        # Reshape.
        batch_size, *_ = s_j.size()
        s_j_norm = s_j.view(batch_size, -1)

        # Calculate.
        s_j_norm = s_j_norm.norm(p=1, dim=1)
        while s_j_norm.dim() < s_j.dim():
            s_j_norm = s_j_norm.view(*s_j_norm.size(), 1)
        s_j_norm_sq = s_j_norm ** 2
        v_j = ((s_j_norm_sq / (1 + s_j_norm_sq)) * (s_j / (s_j_norm + 1e-7)))

        return v_j

    def mul(self, c_ij, u_hat):
        """Multiply the coupling coefficients to the prediction vectors.

        Args:
            c_ij (torch.Tensor): The coupling coefficients.
            u_hat (torch.Tensor): The prediction vectors.

        Returns:
            (torch.Tensor): The total output of the layer.

        Example::
            >>> routing = DynamicRouting(IN_CAPSULES, IN_CHANNELS,
            ...                          OUT_CAPSULES, OUT_CHANNELS,
            ...                          NUM_ROUTING)
            >>> c_ij = torch.Tensor(IN_CAPSULES, OUT_CAPSULES)
            >>> c_ij = Variable(c_ij)
            >>> u_hat = torch.Tensor(BATCH_SIZE,
            ...                      IN_CAPSULES, IN_CHANNELS,
            ...                      OUT_CAPSULES, OUT_CHANNELS,
            ...                      HEIGHT, WIDTH)
            >>> u_hat = Variable(u_hat)
            >>> s_j = torch.Tensor(BATCH_SIZE,
            ...                    OUT_CAPSULES, OUT_CHANNELS,
            ...                    HEIGHT, WIDTH)
            >>> routing.mul(c_ij, u_hat).size() == s_j.size()
            True
        """
        if self.bias is None:
            self._initialize_bias(u_hat)

        batch_size, *_, height, width = u_hat.size()

        # Reshape.
        c_ij = c_ij.view(1, self.in_capsules, 1, self.out_capsules, 1, 1, 1)
        c_ij = c_ij.expand_as(u_hat)

        # Calculate.
        s_j = c_ij * u_hat

        # Reshape.
        s_j = s_j.view(batch_size,
                       (self.in_capsules * self.in_channels),
                       self.out_capsules, self.out_channels,
                       height, width)
        s_j = s_j.sum(dim=1)
        s_j += self.bias

        return s_j

    def dot(self, u_hat, v_j):
        """Calculate inter-layer agreement.

        Args:
            u_hat (torch.Tensor): The prediction vectors.
            v_j (torch.Tensor): The current output of the each capsule in the
                layer.

        Returns:
            (torch.Tensor): The agreement vector.

        Example::
            >>> routing = DynamicRouting(IN_CAPSULES, IN_CHANNELS,
            ...                          OUT_CAPSULES, OUT_CHANNELS,
            ...                          NUM_ROUTING)
            >>> u_hat = torch.Tensor(BATCH_SIZE,
            ...                      IN_CAPSULES, IN_CHANNELS,
            ...                      OUT_CAPSULES, OUT_CHANNELS,
            ...                      HEIGHT, WIDTH)
            >>> v_j = torch.Tensor(BATCH_SIZE,
            ...                    OUT_CAPSULES, OUT_CHANNELS,
            ...                    HEIGHT, WIDTH)
            >>> agreement = torch.Tensor(IN_CAPSULES, OUT_CAPSULES)
            >>> routing.dot(u_hat, v_j).size() == agreement.size()
            True
        """
        batch_size, *_, height, width = u_hat.size()

        # Reshape.
        v_j = v_j.view(batch_size,
                       1, 1,
                       self.out_capsules, self.out_channels,
                       height, width)
        v_j = v_j.expand_as(u_hat)

        # Calculate.
        agreement = u_hat * v_j

        # Reshape.
        agreement = agreement.view(batch_size,
                                   self.in_capsules, self.in_channels,
                                   self.out_capsules, self.out_channels,
                                   (height * width))
        agreement = agreement.sum(dim=5)
        agreement = agreement.sum(dim=4)
        agreement = agreement.sum(dim=2)
        agreement = agreement.sum(dim=0)

        return agreement

    def forward(self, u_hat):
        """Perform dynamic routing and update logits.

        Args:
            u_hat (torch.Tensor): The prediction vector.

        Returns:
            (torch.Tensor): The current output of the each capsule in the
                layer.

        Example::
            >>> routing = DynamicRouting(IN_CAPSULES, IN_CHANNELS,
            ...                          OUT_CAPSULES, OUT_CHANNELS,
            ...                          NUM_ROUTING)
            >>> u_hat = torch.Tensor(BATCH_SIZE,
            ...                      IN_CAPSULES, IN_CHANNELS,
            ...                      OUT_CAPSULES, OUT_CHANNELS,
            ...                      HEIGHT, WIDTH)
            >>> output = torch.Tensor(BATCH_SIZE,
            ...                       OUT_CAPSULES, OUT_CHANNELS,
            ...                       HEIGHT, WIDTH)
            >>> routing.forward(Variable(u_hat)).size() == output.size()
            True
        """
        b_ij = type(u_hat.data)(self.in_capsules, self.out_capsules).fill_(0)

        for _ in range(self.num_routing):
            c_ij = softmax(Variable(b_ij), dim=1)
            s_j = self.mul(c_ij, u_hat)
            v_j = self.squash(s_j)
            b_ij += self.dot(u_hat, v_j).data

        return v_j


class PrimaryCapsules(Module):
    """Perform a convolutional capsule operation.

    Args:
        in_capsules (int): The number of capsules of the input.
        in_channels (int): The number of channels of the input.
        out_capsules (int): The number of capsules of the output.
        out_channels (int): The number of channels of the output.
        num_routing (int): The number of routing logit update iterations.
    """
    def __init__(self, in_capsules, in_channels, out_capsules, out_channels,
                 num_routing):
        super(PrimaryCapsules, self).__init__()

        self.in_capsules = in_capsules
        self.in_channels = in_channels
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.num_routing = num_routing

        self.conv = Conv2d(
            in_channels=(self.in_capsules * self.in_channels),
            out_channels=(self.out_capsules * self.out_channels),
            kernel_size=(9, 9),
            stride=(2, 2),
            bias=False)
        self.routing = DynamicRouting(
            in_capsules=1, in_channels=1,
            out_capsules=out_capsules, out_channels=out_channels,
            num_routing=num_routing)

    def forward(self, x):
        """Apply convolution and perform routing.

        Args:
            x (torch.Tensor): The output of the previous layer.
            (torch.Tensor): The output of the current layer.

        Example::
            >>> primary_capsules = PrimaryCapsules(IN_CAPSULES, IN_CHANNELS,
            ...                                    OUT_CAPSULES, OUT_CHANNELS,
            ...                                    NUM_ROUTING)
            >>> x = torch.Tensor(BATCH_SIZE,
            ...                  IN_CAPSULES, IN_CHANNELS,
            ...                  20, 20)
            >>> x = Variable(x)
            >>> output = torch.Tensor(BATCH_SIZE, OUT_CAPSULES, OUT_CHANNELS,
            ...                       6, 6)
            >>> primary_capsules.forward(x).size() == output.size()
            True
        """
        # Reshape.
        batch_size, *_, height, width = x.size()
        x = x.view(batch_size,
                   (self.in_capsules * self.in_channels),
                   height, width)

        # Calculate.
        x = self.conv(x)

        # Reshape.
        batch_size, *_, height, width = x.size()
        x = x.view(batch_size,
                   1, 1,
                   self.out_capsules, self.out_channels,
                   height, width)

        # Calculate.
        x = self.routing(x)

        return x


class DigitCapsules(Module):
    """Perform a fully-connected capsule operation.

    Args:
        in_capsules (int): The number of capsules of the input.
        in_channels (int): The number of channels of the input.
        out_capsules (int): The number of capsules of the output.
        out_channels (int): The number of channels of the output.
        num_routing (int): The number of routing logit update iterations.
    """
    def __init__(self, in_capsules, in_channels, out_capsules, out_channels,
                 num_routing):
        super(DigitCapsules, self).__init__()

        self.in_capsules = in_capsules
        self.in_channels = in_channels
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.num_routing = num_routing

        self.routing = DynamicRouting(
            in_capsules=in_capsules, in_channels=in_channels,
            out_capsules=out_capsules, out_channels=out_channels,
            num_routing=num_routing)

        self.register_parameter('W', None)

    def _initialize_weights(self, u):
        *_, height, width = u.size()
        W = type(u.data)(self.in_capsules, self.in_channels,
                         self.out_capsules, self.out_channels,
                         height, width)
        W.normal_(std=0.1)
        self.W = Parameter(W)

    def mul(self, u, W):
        """Multiply previous layer output by the weight matrix.

        Args:
            u (torch.Tensor): The output of the previous layer.
            W (torch.Tensor): The weight matrix.

        Returns:
            (torch.Tensor): The prediction vectors.

        Example::
            >>> digit_capsules = DigitCapsules(IN_CAPSULES, IN_CHANNELS,
            ...                                OUT_CAPSULES, OUT_CHANNELS,
            ...                                NUM_ROUTING)
            >>> u = torch.Tensor(BATCH_SIZE,
            ...                  IN_CAPSULES, IN_CHANNELS,
            ...                  HEIGHT, WIDTH)
            >>> W = torch.Tensor(IN_CAPSULES, IN_CHANNELS,
            ...                  OUT_CAPSULES, OUT_CHANNELS,
            ...                  HEIGHT, WIDTH)
            >>> u_hat = torch.Tensor(BATCH_SIZE,
            ...                      IN_CAPSULES, IN_CHANNELS,
            ...                      OUT_CAPSULES, OUT_CHANNELS,
            ...                      HEIGHT, WIDTH)
            >>> digit_capsules.mul(u, W).size() == u_hat.size()
            True
        """
        # Reshape.
        batch_size, *_, height, width = u.size()
        u = u.view(batch_size,
                   self.in_capsules, self.in_channels,
                   1, 1,
                   height, width)
        u = u.expand(batch_size,
                     self.in_capsules, self.in_channels,
                     self.out_capsules, self.out_channels,
                     height, width)
        W = W.view(1, *W.size())
        W = W.expand_as(u)

        # Calculate.
        u_hat = u * W

        return u_hat

    def forward(self, u):
        """Apply weight and perform routing.

        Args:
            u (torch.Tensor): The output of the previous layer.

        Returns:
            (torch.Tensor): The output of the current layer.

        Example::
            >>> digit_capsules = DigitCapsules(IN_CAPSULES, IN_CHANNELS,
            ...                                OUT_CAPSULES, OUT_CHANNELS,
            ...                                NUM_ROUTING)
            >>> u = torch.Tensor(BATCH_SIZE,
            ...                  IN_CAPSULES, IN_CHANNELS,
            ...                  HEIGHT, WIDTH)
            >>> u = Variable(u)
            >>> output = torch.Tensor(BATCH_SIZE,
            ...                       OUT_CAPSULES, OUT_CHANNELS)
            >>> digit_capsules.forward(u).size() == output.size()
            True
        """
        if self.W is None:
            self._initialize_weights(u)

        # Calculate.
        u_hat = self.mul(u, self.W)
        v_j = self.routing(u_hat)

        # Reshape.
        batch_size, *_, height, width = u_hat.size()
        v_j = v_j.view(batch_size,
                       self.out_capsules, self.out_channels,
                       (height * width))
        v_j = v_j.sum(dim=-1)

        return v_j


class CapsNet(Module):
    """Learn and predict MNIST dataset through a simple CapsNet architecture.

    On forward pass, this model outputs a tuple of the input data and the
    the output of digit capsules.
    """
    def __init__(self, in_channels=1, out_capsules=10, out_channels=16):
        super(CapsNet, self).__init__()

        self.bias = Parameter(Tensor(256, 1, 1).fill_(0.1))

        self.conv = Conv2d(in_channels=in_channels,
                           out_channels=256,
                           kernel_size=(9, 9),
                           stride=1,
                           bias=False)
        self.relu = ReLU(inplace=True)

        self.primary_capsules = PrimaryCapsules(in_capsules=1,
                                                in_channels=256,
                                                out_capsules=32,
                                                out_channels=8,
                                                num_routing=1)
        self.digit_capsules = DigitCapsules(in_capsules=32,
                                            in_channels=8,
                                            out_capsules=out_capsules,
                                            out_channels=out_channels,
                                            num_routing=3)

    def forward(self, x):
        """Compute class vectors.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            (torch.Tensor): The class vectors.

        Example:
            >>> capsnet = CapsNet()
            >>> x = torch.Tensor(BATCH_SIZE, 1, 28, 28)
            >>> x = Variable(x)
            >>> y = torch.Tensor(BATCH_SIZE, 10, 16)
            >>> output = capsnet.forward(x)
            >>> output[0] is x
            True
            >>> output[1].size() == y.size()
            True
        """
        y = self.conv(x)
        y = y + self.bias
        y = self.relu(y)

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
