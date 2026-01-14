# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.cortex_m.quantizer.pattern_checkers import (
    CortexMAddMulCheck,
    CortexMConv2DCheck,
    CortexMConvTranspose2DCheck,
    CortexMLinearCheck,
    CortexMSoftmaxCheck,
)

BINARY_OP_PATTERNS = {
    (torch.ops.aten.add.Tensor,): CortexMAddMulCheck,
    (torch.ops.aten.mul.Tensor,): CortexMAddMulCheck,
    (torch.ops.aten.hardswish.default,): CortexMAddMulCheck,  # lowers to mul
    (torch.ops.aten.hardswish_.default,): CortexMAddMulCheck,  # lowers to mul
}

LINEAR_OP_PATTERNS = {
    (torch.ops.aten.linear.default,): CortexMLinearCheck,
    (torch.ops.aten.linear.default, torch.ops.aten.relu.default): CortexMLinearCheck,
    (torch.ops.aten.linear.default, torch.ops.aten.relu_.default): CortexMLinearCheck,
    (
        torch.ops.aten.linear.default,
        torch.ops.aten.hardtanh.default,
    ): CortexMLinearCheck,
    (
        torch.ops.aten.linear.default,
        torch.ops.aten.hardtanh_.default,
    ): CortexMLinearCheck,
    (
        torch.ops.aten.linear.default,
        torch.ops.aten.hardsigmoid.default,
    ): CortexMLinearCheck,
    (
        torch.ops.aten.linear.default,
        torch.ops.aten.hardsigmoid_.default,
    ): CortexMLinearCheck,
    (torch.ops.aten.linear.default, torch.ops.aten.clamp.default): CortexMLinearCheck,
    (torch.ops.aten.linear.default, torch.ops.aten.clamp_.default): CortexMLinearCheck,
}

CONV_OP_PATTERNS = {
    (torch.ops.aten.conv2d.default,): CortexMConv2DCheck,
    (torch.ops.aten.conv2d.default, torch.ops.aten.relu.default): CortexMConv2DCheck,
    (torch.ops.aten.conv2d.default, torch.ops.aten.relu_.default): CortexMConv2DCheck,
    (
        torch.ops.aten.conv2d.default,
        torch.ops.aten.hardtanh.default,
    ): CortexMConv2DCheck,
    (
        torch.ops.aten.conv2d.default,
        torch.ops.aten.hardtanh_.default,
    ): CortexMConv2DCheck,
    (
        torch.ops.aten.conv2d.default,
        torch.ops.aten.hardsigmoid.default,
    ): CortexMConv2DCheck,
    (
        torch.ops.aten.conv2d.default,
        torch.ops.aten.hardsigmoid_.default,
    ): CortexMConv2DCheck,
    (torch.ops.aten.conv2d.default, torch.ops.aten.clamp.default): CortexMConv2DCheck,
    (torch.ops.aten.conv2d.default, torch.ops.aten.clamp_.default): CortexMConv2DCheck,
}

CONV_TRANSPOSE_OP_PATTERNS = {
    (torch.ops.aten.conv_transpose2d.input,): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.relu.default,
    ): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.relu_.default,
    ): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.hardtanh.default,
    ): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.hardtanh_.default,
    ): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.hardsigmoid.default,
    ): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.hardsigmoid_.default,
    ): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.clamp.default,
    ): CortexMConvTranspose2DCheck,
    (
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.clamp_.default,
    ): CortexMConvTranspose2DCheck,
}

SOFTMAX_OP_PATTERNS = {
    (torch.ops.aten._softmax.default,): CortexMSoftmaxCheck,
    (torch.ops.aten.softmax.int,): CortexMSoftmaxCheck,
}

CORTEX_M_QUANTIZER_SUPPORT_DICT = (
    BINARY_OP_PATTERNS
    | LINEAR_OP_PATTERNS
    | CONV_OP_PATTERNS
    | SOFTMAX_OP_PATTERNS
    | CONV_TRANSPOSE_OP_PATTERNS
)
