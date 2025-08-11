# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Final

from executorch.exir.dialects._ops import ops as exir_ops

exir_ops = cast(Any, exir_ops)

qd = exir_ops.edge.quantized_decomposed

QUANT_PER_TENSOR_OP: Final = qd.quantize_per_tensor.default
QUANT_PER_TENSOR_OP_T: Final = qd.quantize_per_tensor.tensor
QUANT_PER_CHANNEL_OP: Final = qd.quantize_per_channel.default

DEQUANT_PER_TENSOR_OP: Final = qd.dequantize_per_tensor.default
DEQUANT_PER_TENSOR_OP_T: Final = qd.dequantize_per_tensor.tensor
DEQUANT_PER_CHANNEL_OP: Final = qd.dequantize_per_channel.default

Q_OPS: Final = (QUANT_PER_TENSOR_OP, QUANT_PER_TENSOR_OP_T, QUANT_PER_CHANNEL_OP)
DQ_OPS: Final = (DEQUANT_PER_TENSOR_OP, DEQUANT_PER_TENSOR_OP_T, DEQUANT_PER_CHANNEL_OP)

PER_TENSOR_QDQ_OPS: Final = (
    QUANT_PER_TENSOR_OP,
    QUANT_PER_TENSOR_OP_T,
    DEQUANT_PER_TENSOR_OP,
    DEQUANT_PER_TENSOR_OP_T,
)
PER_CHANNEL_QDQ_OPS: Final = (QUANT_PER_CHANNEL_OP, DEQUANT_PER_CHANNEL_OP)
