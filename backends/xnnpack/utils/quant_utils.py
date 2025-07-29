# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from itertools import accumulate
from typing import cast, Union

import torch
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)
from torch.fx.experimental.symbolic_shapes import free_symbols, has_free_symbols

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

IS_IMPLICIT_Q_DQ_TAG = "IS_IMPLICIT_Q_DQ_TAG"


def tag_as_implicit_q_dq(node: torch.fx.Node) -> None:
    node.meta[IS_IMPLICIT_Q_DQ_TAG] = True


def is_tagged_as_implicit_q_dq(node: torch.fx.Node) -> bool:
    return node.meta.get(IS_IMPLICIT_Q_DQ_TAG, False)


def is_dynamic_qdq(node: torch.fx.Node) -> bool:
    # check has dynamic qdq name
    if not (is_quant(node) or is_dequant(node)):
        return False

    # check scales and zp are dynamically chosen
    node_input_args = node.args
    if is_affine_qdq(node):
        node_input_args = extract_qdq_affine_op_args_for_decomposed_ops(node)

    scale = node_input_args[1]
    zp = node_input_args[2]
    if not (isinstance(scale, torch.fx.Node) and isinstance(zp, torch.fx.Node)):
        return False

    if not (scale.target == operator.getitem and zp.target == operator.getitem):
        return False

    scale_choose_qparam = scale.all_input_nodes[0]
    zp_choose_qparam = zp.all_input_nodes[0]

    if not (is_qparam(scale_choose_qparam) and is_qparam(zp_choose_qparam)):
        return False

    return True


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


def is_per_tensor(node: torch.fx.Node) -> bool:
    if not (is_quant(node) or is_dequant(node)):
        return False

    is_per_tensor = "per_tensor" in node.target.__name__  # pyre-ignore

    return is_per_tensor and not (is_per_channel(node))


def is_affine_qdq(node: torch.fx.Node) -> bool:
    if not (is_quant(node) or is_dequant(node)):
        return False

    return "quantize_affine" in node.target.__name__  # pyre-ignore


def _get_block_size_input_scale(node: torch.fx.Node):
    assert is_affine_qdq(node)
    block_size = node.args[1]
    input_val = cast(torch.fx.Node, node.args[0]).meta["val"]
    scale_val = cast(torch.fx.Node, node.args[2]).meta["val"]
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

        ic_block_size = block_size[-1]
        if isinstance(ic_block_size, torch.fx.Node):
            ic_block_size = ic_block_size.meta["val"]
            assert free_symbols(
                ic_block_size
            ), f"block_size: {block_size} given, but {block_size[-1]} is not a dynamic symint"

        ic_dim = input_val.shape[-1]
        if isinstance(ic_dim, torch.fx.Node):
            ic_dim = ic_dim.meta["val"]
            assert free_symbols(
                ic_dim
            ), f"input_shape: {input_val.shape} given, but {input_val.shape[-1]} is not a dynamic symint"

        flag &= ic_dim == ic_block_size
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
        # per channel group is only valid on static weights
        # so scales and weights can't have dynamic shape
        if has_free_symbols(input_val.shape) or has_free_symbols(scale_val.shape):
            return False

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

    if is_per_channel_group(node):
        block_sizes = cast(list[int], node.args[1])
        args.append(block_sizes[-1])

    args.append(node.args[-1])

    return args


def is_tensor_subnormal(tensor: torch.Tensor):
    finfo = torch.finfo(tensor.dtype)
    return (tensor >= 0) & (torch.abs(tensor) < finfo.smallest_normal)


def validate_quant_scales(scales: Union[float, torch.Tensor]):
    if isinstance(scales, float):
        scales = torch.tensor([scales])

    is_infinite = torch.isinf(scales) | torch.isnan(scales)

    is_subnormal = is_tensor_subnormal(scales)

    if is_infinite.nonzero().numel() != 0:
        idx = torch.where(is_infinite)
        idx = tuple(int(index[0]) for index in idx)
        value = scales[idx]
        raise ValueError(
            f"Scales must be finite and normal, however found scale value: {value}"
            f" in scale tensor at index: {idx}"
        )

    if is_subnormal.nonzero().numel() != 0:
        idx = torch.where(is_subnormal)
        idx = tuple(int(index[0]) for index in idx)
        value = scales[idx]
        raise ValueError(
            f"Scales must be finite and normal, however found scale value: {value}"
            f" in scale tensor at index: {tuple(idx)}"
        )


def validate_quant_zeropoints(
    zp: Union[float, int, torch.Tensor], dtype: torch.dtype, is_4bit: bool
):
    if not isinstance(zp, torch.Tensor):
        zp = torch.tensor([zp])

    if dtype == torch.int8 or dtype == torch.qint8:
        if is_4bit:
            invalid_zp = (zp < 0) | (zp > 15)
        else:
            invalid_zp = (zp < -128) | (zp > 127)
    elif dtype == torch.uint8 or dtype == torch.quint8:
        invalid_zp = (zp < 0) | (zp > 255)
    elif dtype == torch.int32:
        invalid_zp = zp != 0
    else:
        raise ValueError("Unsupported dtype for quantization")

    if invalid_zp.nonzero().numel() != 0:
        idx = torch.where(invalid_zp)
        idx = tuple(int(index[0]) for index in idx)
        value = zp[tuple(idx)]
        raise ValueError(
            f"Found invalid zeropoint {value}" f" in zero point tensor at index: {idx}"
        )
