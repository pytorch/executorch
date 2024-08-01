# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)

_Q_OPS = {
    "quantize_per_tensor.tensor",
    "quantize_per_tensor.default",
    "quantize_per_channel.default",
    "quantize_per_channel_group.default",
    "quantize_per_token.default",
}

_DQ_OPS = {
    "dequantize_per_tensor.tensor",
    "dequantize_per_tensor.default",
    "dequantize_per_channel.default",
    "dequantize_per_channel_group.default",
    "dequantize_per_token.default",
}


_QPARAM_OPS = {
    "choose_qparams.tensor",
    "choose_qparams_per_token_asymmetric.default",
}

_DYNAMIC_OPS = {
    "quantize_per_tensor.tensor",
    "quantize_per_token.default",
    "dequantize_per_tensor.tensor",
    "dequantize_per_token.default",
}


def is_dynamic_q(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore

    return node_name in _DYNAMIC_OPS


def is_qparam(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore

    return node_name in _QPARAM_OPS


def is_getitem(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False

    return node.target.__name__ == "getitem"  # pyre-ignore


def is_quant(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore

    return node_name in _Q_OPS


def is_dequant(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore

    return node_name in _DQ_OPS


def is_per_channel(node: torch.fx.Node) -> bool:
    if not (is_quant(node) or is_dequant(node)):
        return False

    return "per_channel" in node.target.__name__  # pyre-ignore
