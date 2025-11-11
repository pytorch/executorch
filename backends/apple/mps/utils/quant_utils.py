#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import torch
from executorch.exir.dialects._ops import ops as exir_ops

DQ_GROUP_TARGETS = {
    exir_ops.edge.quantized_decomposed.dequantize_per_channel_group.default,
}

Q_GROUP_TARGETS = {
    exir_ops.edge.quantized_decomposed.quantize_per_channel_group.default,
}

DQ_TARGETS = {
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_token.default,
}.union(DQ_GROUP_TARGETS)

Q_TARGETS = {
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.quantize_per_token.default,
}.union(Q_GROUP_TARGETS)


def is_quant(tensor: torch.fx.Node) -> bool:
    return tensor.target in Q_TARGETS


def is_dequant(tensor: torch.fx.Node) -> bool:
    return tensor.target in DQ_TARGETS


def is_groupwise_q_dq(tensor: torch.fx.Node) -> bool:
    return tensor.target in [DQ_GROUP_TARGETS, Q_GROUP_TARGETS]
