# Copyright (c) 2024-2025 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Collection, Union

import torch

from torch import nn


class Conv1dModule(torch.nn.Module):
    def __init__(
        self,
        bias: bool = True,
        dilation: Union[int, tuple[int, int]] = 1,
        in_channels: int = 4,
        kernel_size: Union[int, tuple[int, int]] = 3,
        out_channels: int = 8,
        padding: Union[str, int, Collection[int]] = 0,
        stride: Union[int, tuple[int, int]] = 2,
        group: int = 1,
    ):
        super().__init__()

        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=group,
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dModule(torch.nn.Module):
    def __init__(
        self,
        bias: bool = True,
        dilation: Union[int, tuple[int, int]] = 1,
        in_channels: int = 4,
        kernel_size: Union[int, tuple[int, int]] = 3,
        out_channels: int = 8,
        padding: Union[str, int, Collection[int]] = 0,
        stride: Union[int, tuple[int, int]] = 2,
        group: int = 1,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=group,
        )

    def forward(self, x):
        return self.conv(x)


class Conv3dModule(torch.nn.Module):
    def __init__(
        self,
        bias: bool = True,
        dilation: Union[int, tuple[int, int]] = 1,
        in_channels: int = 4,
        kernel_size: Union[int, tuple[int, int]] = 3,
        out_channels: int = 8,
        padding: Union[str, int, Collection[int]] = 0,
        stride: Union[int, tuple[int, int]] = 2,
        group: int = 1,
    ):
        super().__init__()

        self.conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=group,
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dAndMaxPool2DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=8, out_channels=32, kernel_size=5, bias=True
        )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.maxpool(x)


class Conv2dConstantPadNDModule(torch.nn.Module):
    def __init__(self, paddings: Collection[int], constant: float | int | None = None):
        super().__init__()
        self.pad = ConstantPadNDModule(paddings, constant)
        self.conv = Conv2dModule()

    def forward(self, x):
        x = self.conv(x)
        return self.pad(x)


class SoftmaxModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


class SoftmaxConvModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.conv = Conv2dModule()
        self.softmax = SoftmaxModule(dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return self.softmax(x)


class ConvWithSigmoid(torch.nn.Module):
    def __init__(self, conv_in_channels: int = 3):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=3,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)


class LinearModule(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=bias)

    def forward(self, x):
        return self.linear(x)


class AddmmModule(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_channels, in_channels))
        self.bias = torch.nn.Parameter(torch.empty(in_channels))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        self.eval()

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight)


class MmModule(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_channels, in_channels))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.eval()

    def forward(self, x):
        return torch.mm(x, self.weight)


class LinearSoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(12, 10)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)

        return x


class ConvFCSoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(4, 64, 2, bias=False)
        self.fc = torch.nn.Linear(1024, 10)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, (-1, 1024))
        x = self.fc(x)
        x = self.softmax(x)

        return x


class ConvFCFCSoftmaxModuleWithoutReshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(4, 5, 2, bias=False)
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class ConstantPadNDModule(torch.nn.Module):
    def __init__(self, paddings: Collection[int], constant: float | int | None = None):
        super().__init__()
        self.paddings = paddings
        self.constant = constant

    def forward(self, x):
        if self.constant is None:
            return torch.nn.functional.pad(x, tuple(self.paddings), "constant")
        else:
            return torch.nn.functional.pad(
                x, tuple(self.paddings), "constant", self.constant
            )


class ConstantPadNDConvModule(torch.nn.Module):
    def __init__(self, paddings: Collection[int], constant: float | int | None = None):
        super().__init__()
        self.pad = ConstantPadNDModule(paddings, constant)
        self.conv = Conv2dModule()

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)


class MaxPool2dModule(torch.nn.Module):
    def __init__(self, padding=0):
        super().__init__()

        self.max_pool2d = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=padding, dilation=1
        )

    def forward(self, x):
        return self.max_pool2d(x)


class MaxPool2dConvModule(torch.nn.Module):
    def __init__(self, padding=0):
        super().__init__()

        self.conv = Conv2dModule()
        self.max_pool2d = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=padding, dilation=1
        )

    def forward(self, x):
        x = self.conv(x)
        return self.max_pool2d(x)


class AvgPool2dModule(torch.nn.Module):
    def __init__(self, count_include_pad, padding=0):
        super().__init__()

        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=padding,
            count_include_pad=count_include_pad,
        )

    def forward(self, x):
        return self.avg_pool(x)


class AvgPool2dConvModule(torch.nn.Module):
    def __init__(self, count_include_pad, padding=0):
        super().__init__()

        self.conv = Conv2dModule()
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=padding,
            count_include_pad=count_include_pad,
        )

    def forward(self, x):
        x = self.conv(x)
        return self.avg_pool(x)


class AdaptiveAvgPool2dModule(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=output_size)

    def forward(self, x):
        return self.adaptive_avg_pool(x)


class AdaptiveAvgPool2dConvModule(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.conv = Conv2dModule(padding=1)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=output_size)

    def forward(self, x):
        x = self.conv(x)
        return self.adaptive_avg_pool(x)


class AdaptiveAvgPool2dConvMeanDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = Conv2dModule()
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_avg_pool(x)
        return x


class ReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class Conv2dWithActivation(torch.nn.Module):
    def __init__(self, activation: torch.nn.Module | Callable, in_channels: int = 3):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3)
        )
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)


class Conv2dReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(4, 64, 2, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class Conv2dPermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 64, 2, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return torch.permute(x, [0, 2, 1, 3])


class Conv2dReLUMaxPoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 2, bias=False)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return self.pool(x)


class AddTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return x + y


class AddTensorConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2dModule(padding=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x + x


class AddTensorOneInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x + x


class SubTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return x - y


class SubTensorConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2dModule(padding=1, stride=1)

    def forward(self, x, y):
        x = self.conv(x)
        return x - y


class SubTensorOneInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x - x


class MeanDimLinearModule(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.linear = torch.nn.Linear(32, 16)

    def forward(self, x):
        x = self.linear(x)
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class MeanDimConvModule(torch.nn.Module):
    def __init__(self, dim, keepdim, out_channels=8):
        super().__init__()
        self.conv = Conv2dModule(stride=1, padding=1, out_channels=out_channels)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        x = self.conv(x)
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


def get_activation(activation, inplace):
    match activation:
        case "relu":
            return nn.ReLU(inplace=inplace)
        case "relu_hardtanh":
            return nn.Hardtanh(inplace=inplace, min_val=0.0, max_val=float("inf"))
        case "relu6":
            return nn.ReLU6(inplace=inplace)
        case "tanh":
            if inplace:
                return torch.tanh
            else:
                return torch.tanh_
        case "sigmoid":
            return nn.Sigmoid()
        case _:
            raise ValueError


class LinearActivationModule(torch.nn.Module):
    def __init__(
        self, activation: str, inplace: bool, in_channels: int, mode: str = "linear"
    ):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in [
            "linear",
            "addmm",
            "mm",
        ], "Mode must be 'linear', 'addmm', or 'mm'"

        if self.mode == "linear":
            self.linear = torch.nn.Linear(in_channels, in_channels)
        else:
            # Manual weight and bias for addmm/mm
            self.weight = torch.nn.Parameter(torch.empty(in_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.empty(in_channels))
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.activation = get_activation(activation, inplace)
        self.eval()

    def forward(self, x):
        if self.mode == "linear":
            x = self.linear(x)
        if self.mode == "addmm":
            x = torch.addmm(self.bias, x, self.weight)
        elif self.mode == "mm":
            x = torch.mm(x, self.weight)
        return self.activation(x)


class ConvActivationModule(torch.nn.Module):
    def __init__(self, activation: str, inplace: bool, in_channels: int):
        super().__init__()

        self.conv = Conv2dModule(in_channels=in_channels)
        self.activation = get_activation(activation, inplace)
        self.eval()

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)
