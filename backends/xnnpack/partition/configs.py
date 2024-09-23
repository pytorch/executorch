# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops

"""
** How to incorporate a new op into the XNNPACK Partitioner? **

[1] When the new edge op being added is direct descendent of a core-aten op,
and is also supported* by XNNPACK, prefer partitioning it via SUPPORTED_OPS
mechanism e.g. torch.add

[2] When the new op being added is not a core-aten op,

[2.1] If the original torch op is supported* by XNNPACK, prefer partitioning it
via SUPPORTED_MODULES. This will require "recomposing" the op before lowering
it to XNNPACK e.g. torch.nn.Linear. Make sure to include all variants of the
modules in the SUPPORTED_MODULES list.

[2.2] If the original torch op is not supported by XNNPACK, then it is assumed
that out of all the decomposed core-aten ops, SUPPORTED_OPS will be lowered to
XNNPACK.

* - Supported fully or partially. The partial support does not mean only few
ops from the decomposition but means only some variants of the op "modes"
possible with the arg combinations.
"""

SUPPORTED_OPS = [
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.floor.default,
    exir_ops.edge.aten.maximum.default,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.constant_pad_nd.default,
    exir_ops.edge.aten.upsample_bilinear2d.default,
    exir_ops.edge.aten.mean.dim,
    exir_ops.edge.aten.max.dim,
    exir_ops.edge.aten.max_pool2d_with_indices.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.ceil.default,
    exir_ops.edge.aten.hardswish.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.pow.Tensor_Scalar,
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten._prelu_kernel.default,
    exir_ops.edge.aten.slice_copy.Tensor,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten._softmax.default,
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.elu.default,
    exir_ops.edge.aten.avg_pool2d.default,
    exir_ops.edge.aten.leaky_relu.default,
    exir_ops.edge.aten.addmm.default,  # TODO(T163877189) add constraint for addmm
]

SUPPORTED_MODULES = [
    torch.nn.Conv1d,
    # TODO(T161981984) recomposed hardswish into a single node
    torch.nn.Hardswish,  # we need to recompose
    torch.nn.Hardsigmoid,  # we can handle decomposition
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm1d,
    torch.nn.Conv2d,
    torch.nn.Linear,
    torch.nn.functional.linear,
    torch.nn.PReLU,  # Without this, the PReLU weight becomes not a get_attr
]

# TODO delete this and should use SUPPORTED_OPS instead once we align fp32 and quant support
SUPPORTED_QUANT_OPS = [
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.mean.dim,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.slice_copy.Tensor,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.mean.dim,
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.max_pool2d_with_indices.default,
    exir_ops.edge.aten.max_pool2d.default,
    exir_ops.edge.aten.constant_pad_nd.default,
    exir_ops.edge.aten.elu.default,
    exir_ops.edge.aten.t_copy.default,
    exir_ops.edge.aten.leaky_relu.default,
    exir_ops.edge.aten.addmm.default,  # TODO(T163877189) add constraint for addmm
]

# This set is used to determine if an op is a supported Quantized Op. This is
# used to determine whether a quantization op is implicit or explicit.
SUPPORTED_IMPLICIT_Q_DQ_OP_NAMES_SET = {
    op.name()
    for op in (
        SUPPORTED_QUANT_OPS
        + [
            exir_ops.edge.aten._to_copy.default,
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.convolution.default,
        ]
    )
}

UNSUPPORTED_QUANT_MODULES = [
    torch.nn.Hardswish,
    torch.nn.Hardsigmoid,
]

# TODO delete this and should use SUPPORTED_MODULES instead once we align fp32 and quant support
SUPPORTED_QUANT_MODULES = [
    torch.nn.Linear,
    torch.nn.functional.linear,
    # TODO - T158982884
    # torch.ao.nn.quantized.reference.modules.linear.Linear,
    torch.nn.Conv1d,
    torch.nn.functional.conv1d,
    torch.ao.nn.quantized.reference.modules.conv.Conv1d,
    torch.nn.Conv2d,
    torch.nn.functional.conv2d,
    torch.ao.nn.quantized.reference.modules.conv.Conv2d,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
]

SUPPORTED_IMPLICIT_Q_DQ_MODULES_SET = set(SUPPORTED_QUANT_MODULES)

# Modules which support dynamic quantization
# These already support dynamic shape.
SUPPORTED_DYN_QUANT_LINEAR_MODULES = [
    torch.nn.Linear,
    torch.nn.functional.linear,
]

SUPPORTED_DYN_QUANT_MODULES = SUPPORTED_DYN_QUANT_LINEAR_MODULES

# XNNPACK supports majority of shape dynamism, however some ops are
# explicitly static, so we maintain a set here to exclude them from
# dynamic shape support.
STATIC_OPS = [
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.slice_copy.Tensor,
]

STATIC_MODULES = []
