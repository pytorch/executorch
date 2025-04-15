# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Collection, Union

import torch


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


class LinearModule(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=bias)

    def forward(self, x):
        return self.linear(x)


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


class ReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


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
