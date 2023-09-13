# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Sample inputs for all core ATen ops
"""
import executorch.exir.dialects.edge.arg.constraints as rel
import torch
from executorch.exir.dialects.edge.arg.model import InArg, InKwarg, Return
from executorch.exir.dialects.edge.arg.type import ArgType

SAMPLE_INPUT = {
    "_log_softmax.default": {  # (Tensor self, int dim, bool half_to_float) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Bool, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "_native_batch_norm_legit_no_training.default": {  # (Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
        "args": [
            InArg(ArgType.Tensor, size=[2, 3, 4, 5]),
            InArg(ArgType.TensorOpt, size=[3]),
            InArg(ArgType.TensorOpt, size=[3]),
            InArg(ArgType.Tensor, size=[3]),
            InArg(ArgType.Tensor, size=[3]),
            InArg(ArgType.Float, value=0.1),
            InArg(ArgType.Float, value=1e-8),
        ],
        "returns": [
            Return(ArgType.Tensor, argname="__ret0", size=[2, 3, 4, 5]),
            Return(ArgType.Tensor, argname="__ret1", size=[0]),
            Return(ArgType.Tensor, argname="__ret2", size=[0]),
        ],
        "broadcast": False,
    },
    "_softmax.default": {  # (Tensor self, int dim, bool half_to_float) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Bool, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "_to_copy.default": {  # (Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InKwarg(
                ArgType.Bool,
                "non_blocking",
                value=False,
                constraints={"values": [False]},
            ),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
            InKwarg(ArgType.MemoryFormat, "memory_format", value=None),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "abs.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "acos.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "acosh.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "add.Tensor": {  # (Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
            InKwarg(ArgType.Scalar, "alpha"),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "add.Scalar": {  # (Tensor self, Scalar other, Scalar alpha=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
            InArg(ArgType.Scalar),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "addmm.default": {  # (Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
            InKwarg(ArgType.Scalar, "beta"),
            InKwarg(ArgType.Scalar, "alpha"),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
        "broadcast_shapes": [
            [[], [6, 2], [2, 3], [6, 3]],
            [[5], [8, 4], [4, 5], [8, 5]],
            [[1, 1], [3, 4], [4, 5], [3, 5]],
            [[6, 1], [6, 2], [2, 4], [6, 4]],
            [[2, 3], [1, 5], [5, 1], [2, 3]],
            [[7, 4], [7, 3], [3, 1], [7, 4]],
            [[3, 1], [1, 7], [7, 2], [3, 2]],
            [[3, 3], [6, 2], [2, 3], [6, 3]],
            [[2, 3], [3, 4], [1, 1], [2, 4]],
        ],
    },
    "alias_copy.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "amax.default": {  # (Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.DimList, value=[0], deps=[0]),
            InArg(ArgType.Keepdim, value=True),
        ],
        "returns": [
            Return(ArgType.Tensor, size=(1, 2)),
        ],
    },
    "amin.default": {  # (Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.DimList, value=[0], deps=[0]),
            InArg(ArgType.Keepdim, value=True),
        ],
        "returns": [
            Return(ArgType.Tensor, size=(1, 2)),
        ],
    },
    "any.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[], dtype=torch.bool),
        ],
    },
    "arange.default": {  # (Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        "args": [
            InArg(ArgType.Scalar, value=1),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [Return(ArgType.Tensor, size=[1])],
    },
    "arange.start_step": {  # (Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        "args": [
            InArg(ArgType.Scalar, value=0),
            InArg(ArgType.Scalar, value=1),
            InArg(ArgType.Scalar, value=1),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[1]),
        ],
    },
    "argmax.default": {  # (Tensor self, int? dim=None, bool keepdim=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.DimOpt, value=1, deps=[0]),
            InArg(ArgType.Keepdim, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2], dtype=torch.long),
        ],
    },
    "argmin.default": {  # (Tensor self, int? dim=None, bool keepdim=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.DimOpt, value=1, deps=[0]),
            InArg(ArgType.Keepdim, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2], dtype=torch.long),
        ],
    },
    "as_strided_copy.default": {  # (Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Shape, value=[4]),
            InArg(ArgType.Stride, value=[1]),
            InArg(ArgType.Param, value=None),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[4]),
        ],
    },
    "asin.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "asinh.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "atan.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "atanh.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "avg_pool2d.default": {  # (Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor, size=[2, 3, 14, 12]),
            InArg(ArgType.Shape, value=[4, 2]),
            InArg(ArgType.Stride, value=[1, 2]),
            InArg(ArgType.LengthList, value=[1, 1]),
            InArg(ArgType.Bool, value=True),
            InArg(ArgType.Bool, value=False),
            InArg(ArgType.Param, value=None),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2, 3, 13, 7]),
        ],
        "broadcast": False,
    },
    "bitwise_and.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
            InArg(ArgType.Scalar, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.bool),
        ],
    },
    "bitwise_and.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
            InArg(ArgType.Tensor, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.bool),
        ],
    },
    "bitwise_not.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.bool),
        ],
    },
    "bitwise_or.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
            InArg(ArgType.Scalar, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.bool),
        ],
    },
    "bitwise_or.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
            InArg(ArgType.Tensor, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.bool),
        ],
    },
    "bitwise_xor.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
            InArg(ArgType.Scalar, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.bool),
        ],
    },
    "bitwise_xor.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
            InArg(ArgType.Tensor, dtype=torch.bool),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.bool),
        ],
    },
    "bmm.default": {  # (Tensor self, Tensor mat2) -> Tensor
        "args": [
            InArg(ArgType.Tensor, size=[1, 2, 2]),
            InArg(ArgType.Tensor, size=[1, 2, 2]),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[1, 2, 2]),
        ],
        "broadcast": False,
    },
    "cat.default": {  # (Tensor[] tensors, int dim=0) -> Tensor
        "args": [
            InArg(
                ArgType.TensorList, value=[InArg(ArgType.Tensor), InArg(ArgType.Tensor)]
            ),
            InArg(ArgType.Dim, value=0, deps=[0]),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[4, 2]),
        ],
    },
    "ceil.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "clamp.default": {  # (Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.ScalarOpt, bounded=True),
            InArg(ArgType.ScalarOpt, bounded=True),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "clone.default": {  # (Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InKwarg(ArgType.MemoryFormat, "memory_format", value=None),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "constant_pad_nd.default": {  # (Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.LengthList, value=[1, 0, 0, 1]),
            InArg(ArgType.Scalar, bounded=True),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[3, 3]),
        ],
    },
    "convolution.default": {  # (Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor
        "args": [
            InArg(ArgType.Tensor, size=[1, 2, 5]),
            InArg(ArgType.Tensor, size=[4, 2, 3]),
            InArg(ArgType.TensorOpt, size=[4]),
            InArg(ArgType.Stride, value=[2]),
            InArg(ArgType.LengthList, value=[2]),
            InArg(ArgType.LengthList, value=[1]),
            InArg(ArgType.Bool, value=False),
            InArg(ArgType.LengthList, value=[0]),
            InArg(ArgType.Param, value=1),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[1, 4, 4]),
        ],
        "broadcast": False,
    },
    "copy.default": {  # (Tensor self, Tensor src, bool non_blocking=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
            InArg(ArgType.Bool, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "cos.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "cosh.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "cumsum.default": {  # (Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "detach_copy.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "div.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "div.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "embedding.default": {  # (Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor, value=[[0, 1], [1, 0]], dtype=torch.long),
            InArg(ArgType.Param, value=-1),
            InArg(ArgType.Bool, value=False),
            InArg(ArgType.Bool, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2, 2, 2]),
        ],
        "broadcast": False,
    },
    "empty.memory_format": {  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
        "args": [
            InArg(ArgType.Shape, value=[2, 2]),
            InKwarg(ArgType.MemoryFormat, "memory_format", value=None),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [
            Return(ArgType.Tensor, fill=3),
        ],
    },
    "eq.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "erf.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "exp.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "expand_copy.default": {  # (Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor, size=[2, 1]),
            InArg(ArgType.Shape, value=[-1, 3]),
            InKwarg(ArgType.Bool, "implicit", value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2, 3]),
        ],
    },
    "fill.Scalar": {  # (Tensor self, Scalar value) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar, bounded=True),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "fill.Tensor": {  # (Tensor self, Tensor value) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.Tensor,
                size=[],
                bounded=True,
                constraints={
                    "rank": lambda deps: 0,
                },
            ),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
        "broadcast": False,
    },
    "floor.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "floor_divide.default": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor, nonzero=True),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "fmod.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor, nonzero=True),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "fmod.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar, value=1),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "full.default": {  # (SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        "args": [
            InArg(ArgType.Shape, value=[2, 2]),
            InArg(ArgType.Scalar, bounded=True),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "full_like.default": {  # (Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar, bounded=True),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
            InKwarg(ArgType.MemoryFormat, "memory_format", value=None),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "ge.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "ge.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "gelu.default": {  # (Tensor self, *, str approximate="none") -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InKwarg(
                ArgType.Param,
                "approximate",
                value="none",
                constraints={"values": ["none", "tanh"]},
            ),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "glu.default": {  # (Tensor self, int dim=-1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[1, 2]),
        ],
    },
    "gt.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "gt.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "hardtanh.default": {  # (Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar, bounded=True),
            InArg(ArgType.Scalar, bounded=True),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "index.Tensor": {  # (Tensor self, Tensor?[] indices) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.TensorOptList,
                value=[
                    InArg(ArgType.Tensor, value=[0, 1]),
                    InArg(ArgType.Tensor, value=[1, 1]),
                ],
                dtype=torch.long,
            ),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2]),
        ],
    },
    "index_put.default": {  # (Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.TensorOptList,
                value=[
                    InArg(ArgType.Tensor, value=[0, 1]),
                    InArg(ArgType.Tensor, value=[1, 1]),
                ],
                dtype=torch.long,
            ),
            InArg(ArgType.Tensor, size=[2]),
            InArg(ArgType.Bool, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
        "broadcast": False,
    },
    "index_select.default": {  # (Tensor self, int dim, Tensor index) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Tensor, value=[1], dtype=torch.long),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[1, 2]),
        ],
        "broadcast": False,
    },
    "isinf.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor, dtype=torch.bool)],
    },
    "isnan.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor, dtype=torch.bool)],
    },
    "le.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "le.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "leaky_relu.default": {  # (Tensor self, Scalar negative_slope=0.01) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "lift_fresh_copy.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "log.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "logical_and.default": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "logical_not.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "logical_or.default": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "logical_xor.default": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "logit.default": {  # (Tensor self, float? eps=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.FloatOpt, value=0.1),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "lt.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "lt.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "masked_fill.Scalar": {  # (Tensor self, Tensor mask, Scalar value) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.Tensor, value=[[True, False], [False, True]], dtype=torch.bool
            ),
            InArg(ArgType.Scalar, bounded=True),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "max.dim": {  # (Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Keepdim, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, argname="values", size=[2]),
            Return(ArgType.Tensor, argname="indices", size=[2], dtype=torch.long),
        ],
    },
    "max_pool2d_with_indices.default": {  # (Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
        "args": [
            InArg(ArgType.Tensor, size=[2, 12, 12]),
            InArg(ArgType.Shape, value=[4, 3]),
            InArg(ArgType.Stride, value=[3, 2]),
            InArg(ArgType.LengthList, value=[2, 1]),
            InArg(ArgType.LengthList, value=[1, 2]),
            InArg(ArgType.Bool, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2, 5, 5]),
            Return(ArgType.Tensor, size=[2, 5, 5], dtype=torch.long),
        ],
        "broadcast": False,
    },
    "mean.dim": {  # (Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.DimListOpt, value=[0], deps=[0]),
            InArg(ArgType.Keepdim, value=False),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2]),
        ],
    },
    "min.dim": {  # (Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Keepdim, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, argname="values", size=[2]),
            Return(ArgType.Tensor, argname="indices", size=[2], dtype=torch.long),
        ],
    },
    "minimum.default": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "mm.default": {  # (Tensor self, Tensor mat2) -> Tensor
        "args": [
            InArg(ArgType.Tensor, constraints={"rank": 2}),
            InArg(
                ArgType.Tensor,
                deps=[0],
                constraints={
                    "rank": 2,
                    "size": lambda deps, d: deps[0].size(1) if d == 0 else None,
                },
            ),
        ],
        "returns": [Return(ArgType.Tensor)],
        "broadcast": False,
    },
    "mul.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "mul.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "native_layer_norm.default": {  # (Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Shape, value=[2]),
            InArg(ArgType.TensorOpt, size=[2]),
            InArg(ArgType.TensorOpt, size=[2]),
            InArg(ArgType.Float, value=1e-5),
        ],
        "returns": [
            Return(ArgType.Tensor, argname="__ret0"),
            Return(ArgType.Tensor, argname="__ret1", size=[2, 1]),
            Return(ArgType.Tensor, argname="__ret2", size=[2, 1]),
        ],
        "broadcast": False,
    },
    "ne.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "ne.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "neg.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "nonzero.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor, value=[[1, 0], [0, 1]]),
        ],
        "returns": [
            Return(ArgType.Tensor, dtype=torch.long),
        ],
    },
    "ones.default": {  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        "args": [
            InArg(ArgType.Shape, value=[2, 2]),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [Return(ArgType.Tensor, fill=3)],
    },
    "permute_copy.default": {  # (Tensor self, int[] dims) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.DimList,
                value=[1, 0],
                deps=[0],
                constraints={"len_min": lambda deps: deps[0].dim()},
            ),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "pow.Tensor_Scalar": {  # (Tensor self, Scalar exponent) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar, value=2),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "pow.Tensor_Tensor": {  # (Tensor self, Tensor exponent) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "reciprocal.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "relu.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "remainder.Tensor": {  # (Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor, nonzero=True),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "remainder.Scalar": {  # (Tensor self, Scalar other) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar, value=1),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "repeat.default": {  # (Tensor self, SymInt[] repeats) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.LengthList,
                value=[1, 2],
                deps=[0],
                constraints={
                    "len_min": lambda deps: deps[0].dim(),
                    "val_min": 0,
                },
            ),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2, 4]),
        ],
    },
    "round.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "rsqrt.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "rsub.Scalar": {  # (Tensor self, Scalar other, Scalar alpha=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
            InArg(ArgType.Scalar),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "scalar_tensor.default": {  # (Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        "args": [
            InArg(ArgType.Scalar, bounded=True),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [Return(ArgType.Tensor, size=[])],
    },
    "scatter_add.default": {  # (Tensor self, int dim, Tensor index, Tensor src) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(
                ArgType.Tensor,
                value=[[0, 1]],
                dtype=torch.long,
                deps=[0, 1, 3],
                constraints={
                    "rank": lambda deps: deps[0].dim(),
                    "size_min": lambda deps, d: 1,
                    "size_max": lambda deps, d: rel.scatter_add_index_size_max(
                        deps[0], deps[1], deps[2], d
                    ),
                },
            ),
            InArg(ArgType.Tensor),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
        "broadcast": False,
    },
    "select_copy.int": {  # (Tensor self, int dim, SymInt index) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Index, value=1, deps=[0, 1]),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2]),
        ],
    },
    "select_scatter.default": {  # (Tensor self, Tensor src, int dim, SymInt index) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.Tensor,
                size=[2],
                deps=[0, 2, 3],
                constraints={
                    "shape": lambda deps: torch.select(deps[0], deps[1], deps[2]).shape,
                },
            ),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Index, value=1, deps=[0, 2]),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
        "broadcast": False,
    },
    "sigmoid.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "sign.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "sin.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "sinh.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "slice_copy.Tensor": {  # (Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.IndexOpt, value=None),
            InArg(ArgType.IndexOpt, value=None),
            InArg(ArgType.Length, value=1),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "slice_scatter.default": {  # (Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.IndexOpt, value=None),
            InArg(ArgType.IndexOpt, value=None),
            InArg(ArgType.Length, value=1),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
        "broadcast": False,
    },
    "split_copy.Tensor": {  # (Tensor self, SymInt split_size, int dim=0) -> Tensor[]
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Length, value=1),
            InArg(ArgType.Dim, value=0, deps=[0]),
        ],
        "returns": [
            Return(
                ArgType.TensorList,
                value=[
                    Return(ArgType.Tensor, size=[1, 2]),
                    Return(ArgType.Tensor, size=[1, 2]),
                ],
            ),
        ],
    },
    "sqrt.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "squeeze_copy.dim": {  # (Tensor self, int dim) -> Tensor
        "args": [
            InArg(ArgType.Tensor, size=[1, 2]),
            InArg(ArgType.Dim, value=0, deps=[0]),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2]),
        ],
    },
    "stack.default": {  # (Tensor[] tensors, int dim=0) -> Tensor
        "args": [
            InArg(
                ArgType.TensorList,
                value=[
                    InArg(ArgType.Tensor),
                    InArg(ArgType.Tensor),
                    InArg(ArgType.Tensor),
                ],
            ),
            InArg(
                ArgType.Dim,
                value=0,
                deps=[0],
            ),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[3, 2, 2]),
        ],
    },
    "sub.Tensor": {  # (Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
            InKwarg(ArgType.Scalar, "alpha"),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "sub.Scalar": {  # (Tensor self, Scalar other, Scalar alpha=1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Scalar),
            InArg(ArgType.Scalar),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "sum.dim_IntList": {  # (Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.DimListOpt, value=[0], deps=[0]),
            InArg(ArgType.Keepdim, value=False),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2]),
        ],
    },
    "t_copy.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "tan.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "tanh.default": {  # (Tensor self) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
        ],
        "returns": [Return(ArgType.Tensor)],
    },
    "transpose_copy.int": {  # (Tensor self, int dim0, int dim1) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
            InArg(ArgType.Dim, value=1, deps=[0]),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "tril.default": {  # (Tensor self, int diagonal=0) -> Tensor
        "args": [
            InArg(ArgType.Tensor, constraints={"rank_min": lambda deps: 2}),
            InArg(ArgType.Index, value=0),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "unbind_copy.int": {  # (Tensor self, int dim=0) -> Tensor[]
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.Dim, value=0, deps=[0]),
        ],
        "returns": [
            Return(
                ArgType.TensorList,
                value=[
                    Return(ArgType.Tensor, size=[2]),
                    Return(ArgType.Tensor, size=[2]),
                ],
            ),
        ],
    },
    "unsqueeze_copy.default": {  # (Tensor self, int dim) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.Dim,
                value=1,
                deps=[0],
                constraints={
                    "val_min": lambda deps: -deps[0].size(deps[1]) - 1,
                    "val_max": lambda deps: deps[0].size(deps[1]),
                },
            ),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2, 1, 2]),
        ],
    },
    "var.dim": {  # (Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(ArgType.DimListOpt, value=[0], deps=[0]),
            InArg(ArgType.Bool, value=True),
            InArg(ArgType.Keepdim, value=False),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[2]),
        ],
    },
    "view_copy.default": {  # (Tensor self, SymInt[] size) -> Tensor
        "args": [
            InArg(ArgType.Tensor),
            InArg(
                ArgType.Shape,
                value=[4],
                deps=[0],
                constraints={
                    "val_prod": lambda deps: torch.prod(
                        torch.Tensor(list(deps[0].shape))
                    )
                },
            ),
        ],
        "returns": [
            Return(ArgType.Tensor, size=[4]),
        ],
    },
    "where.self": {  # (Tensor condition, Tensor self, Tensor other) -> Tensor
        "args": [
            InArg(ArgType.Tensor, dtype=torch.bool),
            InArg(ArgType.Tensor),
            InArg(ArgType.Tensor),
        ],
        "returns": [
            Return(ArgType.Tensor),
        ],
    },
    "zeros.default": {  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        "args": [
            InArg(ArgType.Shape, value=[2, 2]),
            InKwarg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        "returns": [Return(ArgType.Tensor, fill=3)],
    },
}
