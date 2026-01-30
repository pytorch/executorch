# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Operator configs maps a list of operators/operator patterns to a quantization configuration.
These can be used with the OperatorConfigQuantizer to quantize models based on operator patterns.
"""

import torch

from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_CHANNEL_TRANSPOSE_CONFIG,
    INT8_PER_TENSOR_CONFIG,
    SOFTMAX_PER_TENSOR_CONFIG,
)
from torchao.quantization.pt2e.quantizer import OperatorConfig

# ----------------- OPERATOR PATTERN PRESETS -----------------
BINARY_OP_PATTERNS = [
    [torch.ops.aten.add.Tensor],
    [torch.ops.aten.add_.Tensor],
    [torch.ops.aten.mul.Tensor],
    [torch.ops.aten.mul_.Tensor],
    [torch.ops.aten.hardswish.default],
    [torch.ops.aten.hardswish_.default],
]

LINEAR_OP_PATTERNS = [
    [torch.ops.aten.linear.default],
    [torch.ops.aten.linear.default, torch.ops.aten.relu.default],
    [torch.ops.aten.linear.default, torch.ops.aten.relu_.default],
    [torch.ops.aten.linear.default, torch.ops.aten.hardtanh.default],
    [torch.ops.aten.linear.default, torch.ops.aten.hardtanh_.default],
    [torch.ops.aten.linear.default, torch.ops.aten.hardsigmoid.default],
    [torch.ops.aten.linear.default, torch.ops.aten.hardsigmoid_.default],
    [torch.ops.aten.linear.default, torch.ops.aten.clamp.default],
    [torch.ops.aten.linear.default, torch.ops.aten.clamp_.default],
]

CONV_OP_PATTERNS = [
    [torch.ops.aten.conv2d.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.relu.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.relu_.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.hardtanh.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.hardtanh_.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.hardsigmoid.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.hardsigmoid_.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.clamp.default],
    [torch.ops.aten.conv2d.default, torch.ops.aten.clamp_.default],
]

CONV_TRANSPOSE_OP_PATTERNS = [
    [torch.ops.aten.conv_transpose2d.input],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.relu.default],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.relu_.default],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.hardtanh.default],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.hardtanh_.default],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.hardsigmoid.default],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.hardsigmoid_.default],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.clamp.default],
    [torch.ops.aten.conv_transpose2d.input, torch.ops.aten.clamp_.default],
]

SOFTMAX_OP_PATTERNS = [
    [torch.ops.aten._softmax.default],
    [torch.ops.aten.softmax.int],
]

# ----------------- OPERATOR CONFIG PRESETS -----------------
INT8_BINARY_OPS_OPERATOR_CONFIG = OperatorConfig(
    INT8_PER_TENSOR_CONFIG, BINARY_OP_PATTERNS
)

INT8_LINEAR_OPERATOR_CONFIG = OperatorConfig(
    INT8_PER_TENSOR_CONFIG,
    LINEAR_OP_PATTERNS,
)

INT8_CONV_OPERATOR_CONFIG = OperatorConfig(
    INT8_PER_CHANNEL_CONFIG,
    CONV_OP_PATTERNS,
)

INT8_CONV_TRANSPOSE_OPERATOR_CONFIG = OperatorConfig(
    INT8_PER_CHANNEL_TRANSPOSE_CONFIG,
    CONV_TRANSPOSE_OP_PATTERNS,
)

INT8_SOFTMAX_OPERATOR_CONFIG = OperatorConfig(
    SOFTMAX_PER_TENSOR_CONFIG,
    SOFTMAX_OP_PATTERNS,
)
