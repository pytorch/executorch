# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Operator configs maps a list of operators/operator patterns to a quantization configuration.
These can be used with the OperatorConfigQuantizer to quantize models based on operator patterns.
"""

import torch

from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_TENSOR_CONFIG,
)
from torchao.quantization.pt2e.quantizer import OperatorConfig

# ----------------- OPERATOR PATTERN PRESETS -----------------
BINARY_OP_PATTERNS = [
    [torch.ops.aten.add.Tensor],
]

LINEAR_OP_PATTERNS = [
    [torch.ops.aten.linear.default],
    [torch.ops.aten.linear.default, torch.ops.aten.relu.default],
]

# ----------------- OPERATOR CONFIG PRESETS -----------------
INT8_BINARY_OPS_OPERATOR_CONFIG = OperatorConfig(
    INT8_PER_TENSOR_CONFIG, BINARY_OP_PATTERNS
)

INT8_LINEAR_OPERATOR_CONFIG = OperatorConfig(
    INT8_PER_TENSOR_CONFIG,
    LINEAR_OP_PATTERNS,
)
