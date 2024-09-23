# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from itertools import accumulate
from typing import cast

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
    "quantize_affine.default",
}

_DQ_OPS = {
    "dequantize_per_tensor.tensor",
    "dequantize_per_tensor.default",
    "dequantize_per_channel.default",
    "dequantize_per_channel_group.default",
    "dequantize_per_token.default",
    "dequantize_affine.default",
}


_QPARAM_OPS = {
    "choose_qparams.tensor",
    "choose_qparams_per_token_asymmetric.default",
    "choose_qparams_affine.default",
}

_DYNAMIC_OPS = {
    "quantize_per_tensor.tensor",
    "quantize_per_token.default",
    "dequantize_per_tensor.tensor",
    "dequantize_per_token.default",
}


def is_dynamic_qdq(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    is_dynamic_affine = is_per_token(node) and not is_per_channel_group(node)

    return node_name in _DYNAMIC_OPS or is_dynamic_affine


def is_qparam(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore

    return node_name in _QPARAM_OPS


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

    is_affine_per_channel_group = is_per_channel_group(node)
    is_per_channel = "per_channel" in node.target.__name__  # pyre-ignore

    return is_per_channel or is_affine_per_channel_group


def is_affine_qdq(node: torch.fx.Node) -> bool:
    if not (is_quant(node) or is_dequant(node)):
        return False

    return "quantize_affine" in node.target.__name__  # pyre-ignore


def _get_block_size_input_scale(node: torch.fx.Node):
    assert is_affine_qdq(node)
    block_size = node.args[1]
    input_val = node.all_input_nodes[0].meta["val"]
    scale_val = node.all_input_nodes[1].meta["val"]
    return block_size, input_val, scale_val


def is_per_token(node: torch.fx.Node):
    if not (is_quant(node) or is_dequant(node)):
        return False

    if "per_token" in node.target.__name__:  # pyre-ignore
        return True
    elif is_affine_qdq(node):
        block_size, input_val, scale_val = _get_block_size_input_scale(node)
        flag = True
        scale_numel_expected = 1
        for i in range(len(block_size) - 1):
            flag &= block_size[i] == 1
            scale_numel_expected *= input_val.shape[i]

        flag &= block_size[-1] == input_val.shape[-1]
        flag &= scale_val.numel() == scale_numel_expected
        return flag

    return False


def is_per_channel_group(node: torch.fx.Node):
    if not (is_quant(node) or is_dequant(node)):
        return False

    if "per_channel_group" in node.target.__name__:  # pyre-ignore
        return True
    elif is_affine_qdq(node):
        block_size, input_val, scale_val = _get_block_size_input_scale(node)
        flag = True
        flag &= len(block_size) == 2
        flag &= block_size[0] == 1
        group_size = block_size[1]
        scale_numel = list(accumulate(scale_val.shape, operator.mul))[-1]
        input_numel = list(accumulate(input_val.shape, operator.mul))[-1]
        flag &= input_numel == group_size * scale_numel
        return flag

    return False


def extract_qdq_affine_op_args_for_decomposed_ops(node: torch.fx.Node):
    if not is_affine_qdq(node):
        return None, None
    # make sure input_dtype and zero_point_domain have expected values
    input_node = node.args[0]
    scale_node = node.args[2]
    zero_point_node = node.args[3]
    args = [input_node, scale_node, zero_point_node]
    assert (
        len(node.args) > 4
    ), f"expecting at least 6 args, got node: {node.format_node()}"

    if node.args[4] != torch.int8:
        return None, None
    target_dtype = cast(torch.dtype, node.args[4])

    if len(node.args) > 6:
        # quant_min
        args.append(node.args[5])
        # quant_max
        args.append(node.args[6])
    else:
        dtype_info = torch.iinfo(target_dtype)
        quant_min = dtype_info.min
        quant_max = dtype_info.max
        args.append(quant_min)
        args.append(quant_max)

    # add target_dtype_node after quant_min/quant_max
    args.append(target_dtype)
    # zero_point_domain
    if len(node.args) > 7 and node.args[7] != "INT":
        return None, None

    if is_per_channel_group(node):
        block_sizes = cast(list[int], node.args[1])
        args.append(block_sizes[-1])

    args.append(node.args[-1])

    return args
