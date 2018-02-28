"""Define tests for CapsNet.
"""
from unittest import TestCase

from torch import cuda, Tensor, Size, zeros
from torch.autograd import Variable

from capsnet import CapsNet, SeparateMarginLoss

_use_cuda = cuda.is_available()


class TestCapsNet(TestCase):
    """Test CapsNet architecture.
    """
    def setUp(self):
        self.capsnet = CapsNet()

        if _use_cuda:
            self.capsnet = self.capsnet.cuda()

    def test_intermediate_dimensions(self):
        """Test intermediate layer output dimensions.
        """
        x = zeros(1, 1, 28, 28)

        x = Variable(x.cuda())

        x = self.capsnet.conv1(x)
        self.assertEquals(x.size()[1:], Size([256, 20, 20]))

        x = self.capsnet.primary_capsules(x)
        self.assertEquals(x.size()[1:], Size([32, 8, 6, 6]))

        x = self.capsnet.digit_capsules(x)
        self.assertEquals(x.size()[1:], Size([10, 16]))

    def test_result_output_dimensions(self):
        """Test network output dimensions.
        """
        x = zeros(1, 1, 28, 28)

        x = Variable(x.cuda())

        x = self.capsnet(x)
        self.assertEquals(x.size()[1:], Size([10]))


class TestSeparableMarginLoss(TestCase):
    """Test SeparableMarginLoss.
    """
    def test_calculations(self):
        separable_margin_loss = SeparateMarginLoss()

        x = Tensor(1, 10).fill_(0)
        x[0, 0] = 1

        loss = Tensor(1, 10).fill_(0.5 * 0.9 ** 2)
        loss[0, 0] = 0.1

        print(separable_margin_loss(x, x), loss)
        self.assertTrue(separable_margin_loss(x, x).equal(loss))
