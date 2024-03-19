# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops

DQ_TARGETS = {
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_token.default,
}

Q_TARGETS = {
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.quantize_per_channel_group.default,
    exir_ops.edge.quantized_decomposed.quantize_per_token.default,
}


def is_quant(tensor: torch.fx.Node) -> bool:
    return tensor.target in Q_TARGETS


def is_dequant(tensor: torch.fx.Node) -> bool:
    return tensor.target in DQ_TARGETS
