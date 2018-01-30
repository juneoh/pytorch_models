"""A PyTorch implementation of NASNet-A.
"""
import torch

class DepthwiseSeparableConv2d(torch.nn.Module):
    """Define a depthwise separable layer.

    As in the NASNet paper, there is no Batch Normalization and ReLU
    between the depthwise and pointwise convolutions; also, the separable
    convolution is repeated twice.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.sep_conv_1 = self._make_once()
        self.sep_conv_2 = self._make_once()
   
    def _make_once(self):
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            padding=self.padding,
                            groups=self.in_channels,
                            bias=False),
            torch.nn.Conv2d(in_channels=self.out_channels,
                            out_channels=self.out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=self.in_channels))
 
    def forward(self, x):
        x = self.sep_conv_1(x)
        x = self.sep_conv_2(x)

        return x

class Combination(torch.nn.Module):
    """Define a layer that combines the results from two operations.
    """
    def __init__(self, combination=torch.add, operation_1=None,
                 operation_2=None):
        super(Combination, self).__init__()

        self.combination = combination
        self.operation_1 = operation_1
        self.operation_2 = operation_2

    def forward(self, x1, x2):
        if self.operation_1:
            x1 = self.operation_1(x1)

        if self.operation_2:
            x2 = self.operation_2(x2)

        return self.combination(x1, x2)

class FactorizedReduction(torch.nn.Module):
    """Define a layer to reduce shape without information loss due to striding.
    """
    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()

        self.relu = torch.nn.ReLU()

        self.path_1 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=1,
                               stride=2),
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels // 2,
                            kernel_size=1,
                            stride=1,
                            bias=False)
        )

        self.path_2 = torch.nn.Sequential(
            torch.nn.ZeroPad2d([0, 1, 0, 1]),
            torch.nn.AvgPool2d(kernel_size=1,
                               stride=2),
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels // 2,
                            kernel_size=1,
                            stride=1,
                            bias=False),
            torch.nn.ZeroPad2d([-1, 0, -1, 0])
        )

        self.batch_norm = torch.nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.relu(x)

        x_1 = self.path_1(x)
        x_2 = self.path_2(x)

        x = torch.cat([x_1, x_2])
        x = self.batch_norm(x)

        return x

class NormalCell(torch.nn.Module):
    """Define Normal Cell from NASNet-A.
    """
    def __init__(self, in_channels, in_channels_prev, out_channels_prev):
        super(NormalCell, self).__init__()

        self.reduce_prev_layer = None
        if in_channels != in_channels_prev:
            self.reduce_prev_layer = FactorizedReduction(
                in_channels=in_channels_prev,
                out_channels=in_channels
            )

        self.combination_1 = Combination(
            operation_1=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=3,
                                                 padding=1),
            operation_2=None
        )

        self.combination_2 = Combination(
            operation_1=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=3,
                                                 padding=1),
            operation_2=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=5,
                                                 padding=2)
        )

        self.combination_3 = Combination(
            operation_1=torch.nn.AvgPool2d(kernel_size=3, padding=1),
            operation_2=None
        )

        self.combination_4 = Combination(
            operation_1=torch.nn.AvgPool2d(kernel_size=3, padding=1),
            operation_2=torch.nn.AvgPool2d(kernel_size=3, padding=1)
        )

        self.combination_5 = Combination(
            operation_1=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=5,
                                                 padding=2),
            operation_2=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=3,
                                                 padding=1)
        )

    def forward(self, x, x_prev):
        if self.reduce_prev_layer:
            x_prev = self.reduce_prev_layer(x_prev)

        return torch.cat([self.combination_1(x, x),
                          self.combination_2(x_prev, x),
                          self.combination_3(x, x_prev),
                          self.combination_4(x_prev, x_prev),
                          self.combination_5(x_prev, x_prev)])

class ReductionCell(torch.nn.Module):
    """Define Reduction Cell from NASNet-A.
    """
    def __init__(self, in_channels, in_channels_prev):
        super(ReductionCell, self).__init__()

        self.reduce_prev_layer = None
        if in_channels != in_channels_prev:
            self.reduce_prev_layer = FactorizedReduction(
                in_channels=in_channels_prev,
                out_channels=in_channels)

        self.combination_1_1 = Combination(
            operation_1=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels * 2,
                                                 kernel_size=7,
                                                 stride=2,
                                                 padding=3),
            operation_2=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels * 2,
                                                 kernel_size=5,
                                                 stride=2,
                                                 padding=2)
        )

        self.combination_1_2 = Combination(
            operation_1=torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=3,
                                   stride=2,
                                   padding=1),
                torch.nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels * 2,
                                kernel_size=1)
            ),
            operation_2=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels * 2,
                                                 kernel_size=7,
                                                 stride=2,
                                                 padding=3)
        )

        self.combination_1_3 = Combination(
            operation_1=torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=3,
                                   stride=2,
                                   padding=1),
                torch.nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels * 2,
                                kernel_size=1)
            ),
            operation_2=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels * 2,
                                                 kernel_size=5,
                                                 stride=2,
                                                 padding=2)
        )

        self.combination_2_1 = Combination(
            operation_1=torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=3,
                                   stride=2,
                                   padding=2),
                torch.nn.Conv2d(in_channels,
                                in_channels * 2,
                                kernel_size=1)
            ),
            operation_2=DepthwiseSeparableConv2d(in_channels=in_channels,
                                                 out_channels=in_channels,
                                                 kernel_size=3,
                                                 padding=1)
        )

        self.combination_2_2 = Combination(
            operation_1=torch.nn.AvgPool2d(kernel_size=3, padding=1),
            operation_2=None
        )

    def forward(self, x, x_prev):
        if self.reduce_prev_layer:
            x_prev = self.reduce_prev_layer(x_prev)

        x_1_1 = self.combination_1_1(x_prev, x)
        x_1_2 = self.combination_1_2(x, x_prev)
        x_1_3 = self.combination_1_3(x, x_prev)

        x_2_1 = self.combination_2_1(x, x_1_1)
        x_2_2 = self.combination_2_2(x_1_1, x_1_2)

        return torch.cat([x_2_1, x_2_2, x_1_3])

class StemCIFAR(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemCIFAR, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.batch_norm = torch.nn.BatchNorm(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)

        return x

class FinalSoftmax(torch.nn.Module):
    def __init__(self):
        self.relu = torch.nn.ReLU()
        self.global_avg_pool = torch.nn.AvgPool2d()
        self.fc = torch.nn.Linear()

    def forward(self, x):
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)

        return x

class NASNetACIFAR(torch.nn.Module):
    """
    """
    input_shape = [32, 32, 3]
    stem_multiplier = 3

    def __init__(self, cell_repeats=6, num_penult_filters=768):
        super(NASNetACIFAR, self).__init__()

        self.add_module('stem',
                        StemCIFAR(self.input_shape[-1],
                                  num_initial_filters * self.stem_multiplier))

        for i in range(self.cell_repeats):
            self.add_module('normal_cell_1_{}'.format(i + 1),
                            NormalCell(in_channels, in_channels_prev))

        self.add_module('reduction_cell_1',
                        ReductionCell())

        for i in range(self.cell_repeats):
            self.add_module('normal_cell_2_{}'.format(i + 1),
                            NormalCell(in_channels, in_channels_prev))

        self.add_module('reduction_cell_2',
                        ReductionCell())

        for i in range(self.cell_repeats):
            self.add_module('normal_cell_3_{}'.format(i + 1),
                            NormalCell(in_channels, in_channels_prev))

        self.add_module('softmax',
                        FinalSoftmax())

    def forward(self, x):
        for key, module in self._modules:
            if key == 'stem':
                x = module(x)
                x_prev = x

            elif key not in ('stem', 'softmax'):
                _x = module(x, x_prev)
                x, x_prev = _x, x

            elif key == 'softmax':
                x = module(x)

        return x
