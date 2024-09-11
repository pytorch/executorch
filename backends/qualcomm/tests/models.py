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
        return 10 + x


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


class BatchNorm(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.native_batchnorm = torch.nn.BatchNorm2d(n_features)
        self.eval()

    def forward(self, x):
        return self.native_batchnorm(x)


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


class Chunk(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.chunk(x, chunks=2, dim=-1)


class ChunkAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c1, c2 = torch.chunk(x, chunks=2, dim=-1)
        return torch.add(c1, c2)


class Clamp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, max=0)


class CompositeDelegateModule(torch.nn.Module):
    def __init__(
        self,
        compiler_specs,
        partitioner_type,
        capture_method,
        lowered_method,
        quantize_method=None,
    ) -> None:
        super().__init__()
        self.modules = [
            Conv2dSequential(),
            Conv2dSequential(),
            Add(),
            Relu(),
        ]
        self.sample_inputs = [
            (torch.randn([1, 1, 3, 3]),),
            (torch.randn([1, 1, 3, 3]),),
            (torch.randn([1, 2, 3, 3]), torch.randn([1, 2, 3, 3])),
            (torch.randn([1, 2, 3, 3]),),
        ]
        self.lowered_modules = []
        for module, sample_input in zip(self.modules, self.sample_inputs):
            partitioner = partitioner_type(compiler_specs)
            if quantize_method:
                module = quantize_method(module, sample_input)
            edge_prog = capture_method(module, sample_input)
            edge_prog.exported_program = lowered_method(
                edge_prog.exported_program, partitioner
            )
            self.lowered_modules.append(
                edge_prog.exported_program.graph_module._modules.get("lowered_module_0")
            )

    def forward(self, x, y):
        x1 = self.lowered_modules[0](x)
        x2 = self.lowered_modules[1](y)
        x3 = self.lowered_modules[2](x1[0], x2[0])
        x4 = self.lowered_modules[3](x3[0])
        return x4[0]

    def get_random_input(self):
        return (torch.randn([1, 1, 3, 3]), torch.randn([1, 1, 3, 3]))

    def get_reference_module(self):
        class CompositeReferenceModule(torch.nn.Module):
            def __init__(self, modules):
                super().__init__()
                self.modules = modules

            def forward(self, x, y):
                x1 = self.modules[0](x)
                x2 = self.modules[1](y)
                x3 = self.modules[2](x1, x2)
                x4 = self.modules[3](x3)
                return x4

        return CompositeReferenceModule(self.modules)


class ContextBinaryExample(torch.nn.Module):
    def forward(self, x, y):
        x = torch.nn.functional.relu(x)
        y = torch.nn.functional.relu(y)
        return x, y

    def example_inputs(self):
        return {
            "x": torch.randn((1, 3, 3, 3)),
            "y": torch.randn((2, 1, 5, 5)),
        }


class Conv1dSequential(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.first = torch.nn.Conv1d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3),
            padding=1,
            bias=bias,
        )

        self.second = torch.nn.Conv1d(
            in_channels=3,
            out_channels=2,
            kernel_size=(3),
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.second(self.first(x))


# small models
class Conv1dReluLogSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=1
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv(x))
        x = self.logsoftmax(x)
        return x


class Conv2dAvgPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 16, 7, bias=True, stride=2, padding=3, dilation=1
        )
        self.pool = torch.nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.conv(x))


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


class Conv2dSequential(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.first = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
            bias=bias,
        )
        self.second = torch.nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=(3, 3),
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.second(self.first(x))


class Conv2dSingle(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dSumReduceDim(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

    def forward(self, x):
        return torch.sum(self.first(x), dim=(2, 3), keepdim=False)


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


class Gelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        return self.gelu(x)


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


class Index(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx0 = torch.tensor([[0, 1], [2, 3], [4, 5]])
        self.idx1 = torch.tensor([[1, 2], [3, 4], [5, 6]])

    def forward(self, x):
        return x[self.idx0] + x[self.idx1]


class IndexPut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros((1, 1024, 12, 64), dtype=torch.float32),
        )

    def forward(self, input_pos, k_val):
        k_out = torch.ops.aten.index_put_(self.k_cache, [None, input_pos], k_val)
        return k_out


class LayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm([768], eps=1e-6)
        self.linear = torch.nn.Linear(768, 196)

    def forward(self, x):
        return self.linear(self.layer_norm(x))


class LeakyReLUDefault(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.leaky_relu(x)


class LeakyReLUCustom(torch.nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(coeff)

    def forward(self, x):
        return self.leaky_relu(x)


class Linear(torch.nn.Module):
    def __init__(self, use_bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5, use_bias).eval()

    def forward(self, x):
        return self.linear(x)


class LogSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.log_softmax(x, dim=-1)


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


class MulScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._scalar = 3.14

    def forward(self, x):
        out1 = torch.ops.aten.mul.Scalar(x, self._scalar)
        return out1


class MultiheadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = torch.nn.MultiheadAttention(
            96, 12, dropout=0.0, batch_first=True
        )

    def forward(self, x):
        attn_output, _ = self.multi_head_attention(x, x, x, need_weights=False)
        return attn_output


class Pad(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.pad(
            x[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0, mode="constant"
        )


class PixelShuffle(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.pixel_shuffle = torch.nn.PixelShuffle(scale)

    def forward(self, x):
        return self.pixel_shuffle(x)


class PixelUnshuffle(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(scale)

    def forward(self, x):
        return self.pixel_unshuffle(x)


class PixelUnshuffleMathEquivalent(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        b, c, hh, hw = x.size()
        out_channel = c * (self.scale**2)
        h = hh // self.scale
        w = hw // self.scale
        x_view = x.view(b, c, h, self.scale, w, self.scale)
        return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class PowTensorScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(x, 2)


class PReLUDefault(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        return self.prelu(x)


class PReLUPerChannel(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.prelu = torch.nn.PReLU(channels)

    def forward(self, x):
        return self.prelu(x)


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


class ResizeBilinear2D(torch.nn.Module):
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


class ResizeNearest2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output_shape = [dim * 2 for dim in x.shape[-2:]]
        return torch.nn.functional.interpolate(
            x,
            size=list(torch.randn(output_shape).shape),
            mode="nearest",
        )


class RmsNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self.rms = torch.nn.RMSNorm([4], 1e-5)

    def forward(self, x):
        return self.rms(x)


class Rsqrt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rsqrt(x)


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_layer, key_layer, value_layer, attn_mask):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask
        )
        return attn_output


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


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)


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


class SliceCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_ids = torch.randn([1, 512])

    def forward(self, x, y):
        seq_length = y.size()[1]
        return x[:, :seq_length] + self.position_ids[:, :seq_length]


class SliceCopyWithStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_ids = torch.randn([1, 512])
        self.step = 2

    def forward(self, x, y):
        seq_length = y.size()[1]
        return (
            x[:, : seq_length : self.step]
            + self.position_ids[:, : seq_length : self.step]
        )


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
        return x / torch.sqrt(torch.tensor([64.0]))


class Squeeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.stack((x, y))


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


class SumIntList(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, dim=(2, 3), keepdim=True)


class Tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)


class Unbind(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unbind(x)


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
