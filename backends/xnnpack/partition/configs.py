# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops

###
### Module based partitioners
###

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
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.ceil.default,
    exir_ops.edge.aten.hardswish.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.pow.Tensor_Scalar,
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten._prelu_kernel.default,
    exir_ops.edge.aten.slice_copy.Tensor,
]

SUPPORTED_MODULES = [
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.ReLU,
    torch.nn.Sigmoid,
    torch.nn.Softmax,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.Linear,
    torch.nn.functional.linear,
    torch.nn.Hardtanh,
    torch.nn.MaxPool2d,
    torch.nn.LeakyReLU,
    torch.nn.ELU,
    torch.nn.AvgPool2d,
    torch.nn.PReLU,  # Without this, the PReLU weight becomes not a get_attr
    torch.cat,
    torch.concat,
    torch.concatenate,
]

# TODO delete this and should use SUPPORTED_OPS instead once we align fp32 and quant support
SUPPORTED_QUANT_OPS = [
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.mean.dim,
    exir_ops.edge.aten.hardtanh.default,  # TODO - which one module or op or both?
    exir_ops.edge.aten.slice_copy.Tensor,
]

SUPPORTED_IMPLICIT_Q_DQ_OP_NAMES_SET = {
    op.name()
    for op in (
        SUPPORTED_QUANT_OPS
        + [
            exir_ops.edge.aten._to_copy.default,
            exir_ops.edge.aten.max_pool2d.default,
            exir_ops.edge.aten.linear.default,
        ]
    )
}

# TODO delete this and should use SUPPORTED_MODULES instead once we align fp32 and quant support
SUPPORTED_QUANT_MODULES = [
    torch.clamp,
    torch.mean,
    torch.permute,
    torch.permute_copy,
    torch.cat,
    torch.concat,
    torch.concatenate,
    torch.nn.Linear,
    torch.nn.functional.linear,
    # TODO - T158982884
    # torch.ao.nn.quantized.reference.modules.linear.Linear,
    torch.nn.MaxPool2d,
    torch.nn.Conv1d,
    torch.nn.functional.conv1d,
    torch.ao.nn.quantized.reference.modules.conv.Conv1d,
    torch.nn.Conv2d,
    torch.nn.functional.conv2d,
    torch.nn.functional.pad,
    torch.nn.functional.elu,
    torch.ao.nn.quantized.reference.modules.conv.Conv2d,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.ConstantPad2d,
    torch.nn.ELU,
    torch.nn.Hardtanh,
    torch.nn.ReLU,
    torch.nn.functional.relu,
    torch.nn.functional.relu_,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.leaky_relu_,
    torch.nn.LeakyReLU,
    # TODO(): In quant --> export flow source_fn is operator target instead of module name
    # This is actively being fixed, but until, we add these operator target names to partitioenr
    torch.ops.aten.convolution.default,
    torch.ops.aten.addmm.default,
]

SUPPORTED_IMPLICIT_Q_DQ_MODULES_SET = set(SUPPORTED_QUANT_MODULES)

# Modules which support dynamic quantization
SUPPORTED_DYN_QUANT_MODULES = [
    torch.nn.Linear,
    torch.nn.functional.linear,
]
