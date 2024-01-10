# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


# module with related operator only
class Add(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)


class AddConstantFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 10.0 + x


class AddConstantLong(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 10.0 + x


class Arange(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def forward(self, y):
        return torch.arange(self.x, dtype=torch.float32) + y


class AvgPoolModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgPool = torch.nn.AvgPool2d(
            kernel_size=(2, 2),
            padding=(1, 1),
            stride=(1, 1),
            count_include_pad=False,
        )

    def forward(self, x):
        return self.avgPool(x)


class Bmm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class Cast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.type(torch.IntTensor)


class Cat2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.cat((x, y), axis=2)


class Cat3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.concat((y, y, x), axis=2)


class Cat4(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.cat((y, y, x, x), axis=2)


class Ceil(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ceil(x)


class Clamp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, max=0)


class ConvSequential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.second = torch.nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

    def forward(self, x):
        return self.second(self.first(x))


class Div(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.divide(x, y)


class DivConstantFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 10.0


class DivConstantLong(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 10


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 3)

    def forward(self, x):
        return self.embedding(x)


class ExpandCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.expand(3, 4)


class HardSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid = torch.nn.Hardsigmoid()

    def forward(self, x):
        return self.hardsigmoid(x)


class HardSwish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hardswish = torch.nn.Hardswish()

    def forward(self, x):
        return self.hardswish(x)


class HardTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)

    def forward(self, x):
        return self.hardtanh(x)


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5).eval()

    def forward(self, x):
        return self.linear(x)


class MaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            ceil_mode=True,
        )

    def forward(self, x):
        return self.max_pool2d(x)


class MeanWKeppDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, (-1, -2), keepdim=True)


class MeanWOKeppDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, (-1, -2))


class Mul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mul(x, y)


class MulConstantFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 10.0 * x


class MulConstantLong(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 10 * x


class Pad(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.pad(
            x[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0, mode="constant"
        )


class PixelShuffle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_shuffle = torch.nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(x)


class Relu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(1, 12)


class SelectCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)[0, 1, 1:2]


class SliceCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_ids = torch.randn([1, 512])

    def forward(self, x, y):
        seq_length = y.size()[1]
        return x[:, :seq_length] + self.position_ids[:, :seq_length]


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=-1)


class Sqrt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sqrt(x)


class SqrtConstant(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.tensor([64]))


class StaticResizeBilinear2DSizeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output_shape = [dim * 2 for dim in x.shape[-2:]]
        return torch.nn.functional.interpolate(
            x,
            size=list(torch.randn(output_shape).shape),
            mode="bilinear",
            align_corners=False,
        )


class Sub(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


class SubConstantFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 10.0 - x


class SubConstantLong(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 10 - x


class Tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)


class Unsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(0)


class View(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_size = 2
        self.second_size = 256

    def forward(self, x, y):
        new_shape = x.size()[:-1] + (self.first_size, self.second_size)
        return x.view(new_shape)


# small models
class Conv2dBnHardtanhMean(torch.nn.Module):
    def __init__(self):
        super(Conv2dBnHardtanhMean, self).__init__()
        groups = 1
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 1
        out_channels = 1

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        self.conv.weight = torch.nn.Parameter(torch.randn(self.conv.weight.size()))
        self.native_batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)
        self.eval()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.native_batchnorm(x1)
        x3 = self.hardtanh(x2)
        x4 = torch.mean(x3, (1), keepdim=True)
        return x4


class Conv2dAvgPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 16, 7, bias=True, stride=2, padding=3, dilation=1
        )
        self.pool = torch.nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.conv(x))


class Conv2dMaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(1, 1),
            padding=1,
            bias=True,
        )
        self.pool = torch.nn.MaxPool2d(1, 1)

    def forward(self, x):
        return self.pool(self.conv(x))


class Conv2dCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        z = torch.cat([x, y], dim=1)
        return z


class ResidualBlockModule(torch.nn.Module):
    def __init__(self):
        super(ResidualBlockModule, self).__init__()
        groups = 1
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 32
        out_channels = 32

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        self.native_batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6.0)
        self.eval()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.native_batchnorm(x1)
        x3 = self.conv(x2)
        x4 = self.native_batchnorm(x3)
        x5 = self.hardtanh(x4)
        x6 = torch.add(x5, x2)
        return x6


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel_sz = 32
        self.conv1 = torch.nn.Conv2d(kernel_sz, kernel_sz, 3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(kernel_sz, kernel_sz, 3, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(kernel_sz, kernel_sz, 3, padding=1, bias=False)
        self.conv4 = torch.nn.Conv2d(kernel_sz, kernel_sz, 3, padding=1, bias=False)
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm2d(kernel_sz)
        self.add = torch.add
        self.mean = torch.mean
        self.reshape = torch.reshape
        self.linear = torch.nn.Linear(4, 10)
        self.permute = torch.permute
        self.eval()

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.batch_norm(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.relu(x4)
        y1 = self.conv3(y)
        y2 = self.batch_norm(y1)
        y3 = self.relu(y2)
        y4 = self.conv4(y3)
        y5 = self.relu(y4)
        z = self.add(x5, y5)
        z1 = self.permute(z, (0, 3, 2, 1))
        z2 = torch.mean(z1, [1, 2], True)
        z3 = self.reshape(z2, (8, -1))
        z4 = self.linear(z3)
        z5 = self.hardtanh(z4)
        return z5


class ViewPermuteMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_size = 2
        self.second_size = 256

    def forward(self, x, y):
        new_shape = x.size()[:-1] + (self.first_size, self.second_size)
        x = x.view(new_shape)
        x = x.permute(0, 2, 1, 3)
        return torch.matmul(x, y.transpose(-1, -2))
