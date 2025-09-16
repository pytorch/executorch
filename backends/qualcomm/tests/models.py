# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


# module with related operator only


# Ensure alias_copy is removed in remove_redundancy pass
class Alias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        alias_x = torch.ops.aten.alias.default(x)
        return self.relu(alias_x)


class And(torch.nn.Module):
    def __init__(self, pos, neg):
        super().__init__()
        self.pos = pos
        self.neg = neg

    def forward(self, x, y):
        bitwise_and = torch.bitwise_and(x, y).bool()
        return torch.where(bitwise_and, self.pos, self.neg)


class Abs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class AdaptiveAvgPool1D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        adaptive_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        return adaptive_avg_pool(x)


class AdaptiveAvgPool2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        return adaptive_avg_pool(x)


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


class Any(torch.nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.any(x, dim=self.dim, keepdim=self.keepdim)


class AMax(torch.nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.amax(x, dim=self.dim, keepdim=self.keepdim)


class AMaxFollowingConv2D(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dim=None, keepdim=False
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        x = self.conv(
            x
        )  # Apply convolution (output shape: [batch, out_channels, H, W])
        return torch.amax(x, dim=self.dim, keepdim=self.keepdim)


class AMin(torch.nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.amin(x, dim=self.dim, keepdim=self.keepdim)


class Arange(torch.nn.Module):
    def __init__(self, start, end, step, dtype):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.dtype = dtype

    def forward(self, y):
        return (
            torch.arange(
                start=self.start, end=self.end, step=self.step, dtype=self.dtype
            )
            + y
        )


class Argmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.argmax(x, dim=0, keepdim=True)
        return x


class Argmin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.argmin(x, dim=0, keepdim=True)
        return x


class ArgminViewSqueezeConv2D(torch.nn.Module):
    def __init__(self):
        # This model is mainly to test the PASS I64toI32
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, y):
        argmin_out = torch.argmin(x, dim=0, keepdim=True)
        index_out = y[argmin_out]
        conv_out = self.conv(index_out)

        view_out = argmin_out.view(-1)
        squeeze_out = view_out.squeeze(-1)
        return squeeze_out, conv_out


class Asin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asin(x)


class Atan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.atan(x)


class AvgPoolModule(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding, ceil_mode):
        super().__init__()
        self.avgPool = torch.nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=False,
        )

    def forward(self, x):
        return self.avgPool(x)


class BatchNorm(torch.nn.Module):
    def __init__(self, n_features, affine=True):
        super().__init__()
        self.native_batchnorm = torch.nn.BatchNorm2d(n_features, affine=affine)
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


class CastMultiUsers(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        index = x.to(torch.long)
        res = torch.gather(y, dim=1, index=index)
        return res + index.to(torch.int32)


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


class CausalMask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("causal_mask", torch.zeros((1, 1, 1, 128)))
        self.mask_length = 128

    def forward(self, padding_mask):
        self.causal_mask[:, :, :, : self.mask_length] = self.causal_mask[
            :, :, :, : self.mask_length
        ].masked_fill(padding_mask, 1)
        return self.causal_mask + 1


class CDist(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.cdist(x, y, p=2)


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


class ClampMax(torch.nn.Module):
    def __init__(self, max):
        super().__init__()
        self.max = max

    def forward(self, x):
        return torch.clamp_max(x, max=self.max)


class ClampMin(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.min = min

    def forward(self, x):
        return torch.clamp_min(x, min=self.min)


class CompositeDelegateModule(torch.nn.Module):
    def __init__(
        self,
        compiler_specs,
        to_edge_transform_and_lower_method,
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
            if quantize_method:
                module = quantize_method(module, sample_input)
            edge_prog = to_edge_transform_and_lower_method(
                module, sample_input, compiler_specs
            )
            self.lowered_modules.append(
                edge_prog.exported_program().graph_module._modules.get(
                    "lowered_module_0"
                )
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
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=1
        )
        self.logsoftmax = torch.nn.LogSoftmax(dim=dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv(x))
        x = self.logsoftmax(x)
        return x


class Conv2dArgmin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            3, 16, 7, bias=True, stride=2, padding=3, dilation=1
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.argmin(x, dim=0, keepdim=True)


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
    def __init__(self, bias=True, channel_last=False):
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
        self.channel_last = channel_last

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last) if self.channel_last else x
        return self.second(self.first(x))


class Conv2dSingle(torch.nn.Module):
    def __init__(
        self,
        bias=True,
        in_channel=1,
        out_channel=3,
        kernel_size=(3, 3),
        padding=1,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranspose1dSingle(torch.nn.Module):
    def __init__(self, bias=True, dilation=1):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv_transpose(x)


class ConvTranspose2dSingle(torch.nn.Module):
    def __init__(self, bias=True, dilation=1):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv_transpose(x)


class Conv2dDownUpSample(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
        )
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.conv_transpose(self.conv(x))


class Conv2dFlip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.dims = [1, 3]

    def forward(self, x):
        x = self.conv(x)
        return torch.flip(x, self.dims)


class Conv2dSliceCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, 2:, :, :]


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


class Conv2dTopK(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)

    def forward(self, x):
        x = self.conv(x)
        topk_values, topk_indices = torch.topk(x, 5, dim=1)
        return topk_values


class Cos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)


class CumSum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.cumsum(dim=0)


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


class DrawGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        kernel_sz = 32
        self.conv1 = torch.nn.Conv2d(kernel_sz, kernel_sz, 3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(kernel_sz, kernel_sz, 3, padding=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.relu1(x1)
        y2 = self.relu1(x2)
        return y1 + y2


class EinsumBilinear(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bn, anm, bm):
        return torch.einsum("bn,anm,bm->ba", bn, anm, bm)


class EinsumOuterProduct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i, j):
        return torch.einsum("i,j->ij", i, j)


class EinsumOuterProductRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i, j):
        return torch.relu(torch.einsum("i,j->ij", i, j))


class Elu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = torch.nn.ELU(alpha=0.5)

    def forward(self, i):
        return self.elu(i)


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 3)

    def forward(self, x):
        return self.embedding(x)


class Equal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x == y


class EqualConstant(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x == self.constant


class ExpandCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.expand(3, 4)


class ExpandAs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.linalg.vector_norm(x)
        y = torch.clamp_min(y, min=1e-10)
        return y.expand_as(x)


class ExpM1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.special.expm1(x)


class Flip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dims = [0, 2]

    def forward(self, x):
        return torch.flip(x, self.dims)


class Floor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.floor(x)


class FloorDiv(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.floor_divide(x, y)


class FloorDivConstantFloat(torch.nn.Module):
    def __init__(self, constant=2.0):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return torch.floor(x / self.constant)


class Fold(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = (32, 32)
        self.patch_height = 2
        self.patch_width = 2

    def forward(self, x):
        fold = torch.nn.functional.fold(
            x,
            output_size=self.output_size,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        return fold


class Full(torch.nn.Module):
    def __init__(self, fill, shape):
        super().__init__()
        self.fill = fill
        self.shape = shape

    def forward(self, x):
        return torch.min(x, torch.full(self.shape, self.fill))


class FullLike(torch.nn.Module):
    def __init__(self, fill):
        super().__init__()
        self.fill = fill

    def forward(self, x):
        return torch.min(x, torch.full_like(x, self.fill))


class Gather(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.gather(x, dim=1, index=y)


class GatherArgmin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        indice = torch.argmin(x, dim=1, keepdim=True)
        return torch.gather(x, dim=1, index=indice)


class GatherWhere(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        index = torch.where(y > 0, torch.Tensor([1]).int(), torch.Tensor([1]).int()).to(
            torch.int64
        )
        return torch.gather(x, x.dim() - 1, index)


class Gelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        return self.gelu(x)


class GreaterEqual(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x >= y


class GreaterThan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x > y


class GreaterEqualConstant(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x >= self.constant


class GreaterThanConstant(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x > self.constant


class GroupNorm(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            32,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.norm = torch.nn.GroupNorm(32, 256)

    def forward(self, x):
        y = self.conv(x)
        return y, self.norm(y)


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
    def __init__(self, axis):
        super().__init__()
        self.idx0 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
        self.idx1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
        self.axis = axis
        self.dispatcher = {
            0: lambda x: x[self.idx0] + x[self.idx1],
            1: lambda x: x[:, self.idx0] + x[:, self.idx1],
            2: lambda x: x[:, :, self.idx0] + x[:, :, self.idx1],
        }

    def forward(self, x):
        return self.dispatcher[self.axis](x)


class IndexCopy(torch.nn.Module):
    def __init__(self, copy_dim=1, skip_mutable_buffer=False):
        super().__init__()
        self.skip_mutable_buffer = skip_mutable_buffer
        self.copy_dim = copy_dim
        self.register_buffer(
            "k_cache",
            torch.zeros((1, 1024, 12, 64), dtype=torch.float32),
            persistent=True,
        )

    def forward(self, input_pos, k_val):
        k_out = self.k_cache
        k_out.index_copy_(self.copy_dim, input_pos, k_val)
        return k_out + 0


class IndexPut(torch.nn.Module):
    def __init__(self, skip_mutable_buffer=False):
        super().__init__()
        self.skip_mutable_buffer = skip_mutable_buffer
        self.register_buffer(
            "k_cache",
            torch.zeros((1, 1024, 12, 64), dtype=torch.float32),
            persistent=True,
        )

    def forward(self, input_pos, k_val):
        k_out = torch.ops.aten.index_put_(self.k_cache, [None, input_pos], k_val)
        return k_out + 0


class IndexSelect(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, indices):
        return torch.index_select(x, self.dim, indices)


class InstanceNorm2d(torch.nn.Module):
    def __init__(self, n_features, affine=True):
        super().__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(n_features, affine=affine)
        self.eval()

    def forward(self, x):
        return self.instance_norm(x)


class LargeTensorLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 4096
        self.linear1 = torch.nn.Linear(512, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 512)

    def forward(self, x):
        x1 = self.linear1(x) + self.linear1(x)
        return self.linear2(x1)


class LayerNorm(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm([768], eps=1e-6, bias=bias)
        self.linear = torch.nn.Linear(768, 196)

    def forward(self, x):
        return self.linear(self.layer_norm(x))


class LayerNormAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm([512], eps=1e-6, bias=False)

    def forward(self, x, y):
        return self.layer_norm(x) + y


class LeakyReLUDefault(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.leaky_relu(x)


class LeakyReLUCustom(torch.nn.Module):
    def __init__(self, coeff, inplace=False):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(coeff, inplace=inplace)

    def forward(self, x):
        return self.leaky_relu(x)


class LessEqual(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x <= y


class LessThan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x < y


class LessEqualConstant(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x <= self.constant


class LessThanConstant(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return self.constant < x


class LiftAddTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N = 2 - 1
        return x + N


class Linear(torch.nn.Module):
    def __init__(self, use_bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(512, 32, use_bias).eval()

    def forward(self, x):
        return self.linear(x)


class LinalgVectorNorm(torch.nn.Module):
    def __init__(self, ord=2.0, dim=None, keepdim=False):
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.linalg.vector_norm(
            x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
        )


class Log(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)


class LogicalAnd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.logical_and(x != 0, y != 0).float()


class LogicalNot(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.logical_not(x > 0)


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


class MaskedFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_mask):
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )


class MaskedSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_mask, input):
        attn_weights = torch.where(
            attention_mask == 0, input, torch.amin(input, dim=3, keepdim=True) + (-20)
        )
        return torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)


class MaxDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        max_logits, max_indices = torch.max(logits, dim=1)
        return max_logits, max_indices


class Maximum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.maximum(x, y)


class MinDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        min_logits, min_indices = torch.min(logits, dim=1)
        return min_logits, min_indices


class Minimum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.minimum(x, y)


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


class Neg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.neg(x)


class NotEqual(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x != y


class NotEqualConstant(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x != self.constant


class OrBitWise(torch.nn.Module):
    def __init__(self, pos, neg):
        super().__init__()
        self.pos = pos
        self.neg = neg

    def forward(self, x, y):
        bitwise_or = torch.bitwise_or(x, y).bool()
        return torch.where(bitwise_or, self.pos, self.neg)


class OrOperator(torch.nn.Module):
    def __init__(self, pos, neg):
        super().__init__()
        self.pos = pos
        self.neg = neg

    def forward(self, x, y):
        operator_or = x.to(torch.bool) | y.to(torch.bool)
        return torch.where(operator_or, self.pos, self.neg)


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


class Relu6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()

    def forward(self, x):
        return self.relu6(x)


class Repeat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.repeat(1, 2, 3, 4)


class ReWriteObs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.relu(x).expand(3, 4)


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


class ResizeBicubic(torch.nn.Module):
    def __init__(self, size, scale_factor, align_corners):
        super().__init__()
        self.align_corners = align_corners
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=self.align_corners,
        )


class ResizeBilinear2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output_shape = [dim * 2 for dim in x.shape[-2:]]
        return torch.nn.functional.interpolate(
            x,
            size=output_shape,
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
            size=output_shape,
            mode="nearest",
        )


class UpsampleNearest2D(torch.nn.Module):
    def __init__(self, sizes=None, scale_factor=None):
        super().__init__()
        self.upsample_neareast_2d = torch.nn.UpsamplingNearest2d(  # noqa: TOR101
            size=sizes, scale_factor=scale_factor
        )

    def forward(self, x):
        return self.upsample_neareast_2d(x)


class RmsNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self.rms = torch.nn.RMSNorm([4], 1e-5)

    def forward(self, x):
        return self.rms(x)


class Roll(torch.nn.Module):
    def __init__(self, shifts, dims=None):
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, x):
        return torch.roll(x, shifts=self.shifts, dims=self.dims)


class Round(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.round(x)


class Rsqrt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rsqrt(x)


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def forward(self, query_layer, key_layer, value_layer, attn_mask):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask, scale=self.scale
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


class Sign(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sign(x)


class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


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


class SliceCopyDefaultParameter(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([x[:1], x[1:]], dim=1)


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


class SliceScatter(torch.nn.Module):
    def __init__(self, dim, start, end, step):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x, y):
        return x.slice_scatter(
            y, dim=self.dim, start=self.start, end=self.end, step=self.step
        )


class Softmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=self.dim)


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


class SquaredReLU(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class Squeeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.stack((x, y, z))


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


class SimpleSubModules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = Add()
        self.sub = Sub()

    def forward(self, a, b, c, d):
        lhs = self.add(a, b)
        rhs = self.sub(c, d)
        return torch.mul(lhs, rhs)


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


class TopKandIndex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx_source = torch.rand(10, 3)

    def forward(self, x):
        a, b = torch.topk(x, 3)
        return a + self.idx_source[b]


class Unbind(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unbind(x)


class Unfold(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_height = 2
        self.patch_width = 2

    def forward(self, x):
        unfold = torch.nn.functional.unfold(
            x,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        return unfold


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


class Where(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.where(x >= torch.zeros(x.shape), y, z)


class WhereConstant(torch.nn.Module):
    def __init__(self, pos, neg):
        super().__init__()
        self.register_buffer("pos", pos)
        self.register_buffer("neg", neg)

    def forward(self, x):
        return torch.where(x >= torch.zeros(x.shape), self.pos, self.neg)


class WhereConstantOther(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x >= 0, torch.ones(x.shape), 0)


class WhereConstantAll(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x >= 0, 1, 0)


class WhereConstantInf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.softmax(
            torch.where(x >= 0, 0.1, float("-inf")), dim=-1
        )


class XorBitWise(torch.nn.Module):
    def __init__(self, pos, neg):
        super().__init__()
        self.pos = pos
        self.neg = neg

    def forward(self, x, y):
        bitwise_xor = torch.bitwise_xor(x, y).bool()
        return torch.where(bitwise_xor, self.pos, self.neg)


class XorOperator(torch.nn.Module):
    def __init__(self, pos, neg):
        super().__init__()
        self.pos = pos
        self.neg = neg

    def forward(self, x, y):
        operator_xor = x.to(torch.bool) ^ y.to(torch.bool)
        return torch.where(operator_xor, self.pos, self.neg)


# Mimi Decoder has 0D tensor which QNN cannot handle.
class ZeroDimTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        input1 = torch.zeros(1)
        selected_element = torch.select(input1, 0, 0)
        return torch.add(x, selected_element)
