#!/usr/bin/env python3
"""Define tests for CapsNet.
"""
from unittest import TestCase

import torch
from torch import DoubleTensor, Size
from torch.autograd import Variable

from capsnet import (CapsNet, CapsNetLoss, ReconstructionLoss,
                     SeparateMarginLoss)

BATCH_SIZE = 3
IN_CHANNELS = 1
IN_WIDTH = 28
IN_HEIGHT = 28
OUT_CAPSULES = 10
OUT_CHANNELS = 16


class TestCapsNet(TestCase):
    """Test CapsNet architecture.
    """
    def test_dimensions(self):
        """Test dimensions of layer outputs.
        """
        capsnet = CapsNet()

        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, IN_WIDTH, IN_HEIGHT)
        x = Variable(x)

        x = capsnet.conv1(x)
        self.assertEquals(x.size(), Size([BATCH_SIZE, 256, 20, 20]))

        x = capsnet.primary_capsules(x)
        self.assertEquals(x.size(), Size([BATCH_SIZE, 32, 8, 6, 6]))

        x = capsnet.digit_capsules(x)
        self.assertEquals(x.size(), Size([BATCH_SIZE,
                                          OUT_CAPSULES,
                                          OUT_CHANNELS]))

    def test_forward_backward(self):
        """Test for forward and backward.
        """
        for use_cuda in (False, True):
            capsnet = CapsNet()
            criterion = CapsNetLoss()

            x = torch.zeros(BATCH_SIZE, IN_CHANNELS, IN_WIDTH, IN_HEIGHT)
            x = Variable(x, requires_grad=True)
            y = torch.zeros(BATCH_SIZE)
            y = Variable(y)

            if use_cuda:
                capsnet = capsnet.cuda()
                criterion = criterion.cuda()
                x = x.cuda()
                y = y.cuda()

            output = capsnet(x)
            loss = criterion(output, y)

            loss.backward()


class TestSeparableMarginLoss(TestCase):
    """Test SeparableMarginLoss.
    """
    def test_calculations(self):
        """Test loss calculation.

        torch.DoubleTensor is used instead of torch.Tensor, which is
        torch.FloatTensor by default, to match the precision with Python's
        float type.
        """
        separable_margin_loss = SeparateMarginLoss()

        x = torch.zeros(2, 10).type(DoubleTensor)
        x[0, 0] = 1
        x[1, 0] = 1

        y = DoubleTensor([0, 1])

        loss = separable_margin_loss(Variable(x), Variable(y))

        self.assertEquals(loss.data[0], 0)
        self.assertEquals(loss.data[1], ((0.5 * 0.9 ** 2) + (0.9 ** 2)))


class TestReconstructionLoss(TestCase):
    """Test ReconstructionLoss.
    """
    def test_dimensions(self):
        """Test dimensions of output.
        """
        reconstruction_loss = ReconstructionLoss(
            in_capsules=OUT_CAPSULES,
            in_channels=OUT_CHANNELS,
            out_channels=IN_CHANNELS * IN_WIDTH * IN_HEIGHT)

        x = torch.zeros(BATCH_SIZE, OUT_CAPSULES, OUT_CHANNELS)
        y = torch.zeros(BATCH_SIZE, IN_CHANNELS * IN_WIDTH * IN_HEIGHT)
        loss = reconstruction_loss(Variable(x), Variable(y))

        self.assertEquals(loss.size(), Size([1]))


class TestCapsNetLoss(TestCase):
    """Test CapsNetLoss.
    """
    def test_dimensions(self):
        """Test dimensions of output.
        """
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, IN_WIDTH, IN_HEIGHT)
        output = torch.zeros(BATCH_SIZE, OUT_CAPSULES, OUT_CHANNELS)
        y = torch.zeros(BATCH_SIZE)

        capsnet_loss = CapsNetLoss()
        loss = capsnet_loss((Variable(x), Variable(output)), Variable(y))

        self.assertEquals(loss.size(), Size([1]))
