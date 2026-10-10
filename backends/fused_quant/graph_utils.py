# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
from executorch.backends.fused_quant.ops import QuantParamsStruct
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import get_constant
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch import fx
from torch._ops import OpOverload
from torch.export import ExportedProgram


_OpOverload = EdgeOpOverload | OpOverload
_QPARAM_FIELDS = ("scale", "zero_point", "dtype", "quant_min", "quant_max")


def split_fused_arg_names(target: _OpOverload) -> tuple[list[str], list[str]]:
    """Split a fused op's schema into quantized inputs and extra arguments."""
    names = [argument.name for argument in target._schema.arguments]
    try:
        first_qparam = next(
            index for index, name in enumerate(names) if name.endswith("_scale")
        )
    except StopIteration as error:
        raise ValueError(f"Fused op {target} has no qparam blocks") from error

    input_names = names[:first_qparam]
    cursor = first_qparam
    for prefix in (*input_names, "out"):
        expected = [f"{prefix}_{field}" for field in _QPARAM_FIELDS]
        actual = names[cursor : cursor + len(_QPARAM_FIELDS)]
        if actual != expected:
            raise ValueError(
                f"Fused op {target} does not follow the qparam argument convention: "
                f"expected {expected}, got {actual}"
            )
        cursor += len(_QPARAM_FIELDS)
    return input_names, names[cursor:]


def is_quantize_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.torchao.quantize_affine.default,
    )


def is_dequantize_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.torchao.dequantize_affine.default,
    )


def is_affine_quant_node(node: fx.Node) -> bool:
    """Check if a node is a torchao affine quantize/dequantize op.

    Affine nodes carry an explicit ``block_size`` and a full-rank scale/zero_point
    whose shape already encodes the block layout -- exactly QuantParamsStruct's
    convention, covering per-tensor/channel/group/blockwise uniformly with no axis.
    """
    return node.target in (
        torch.ops.torchao.quantize_affine.default,
        torch.ops.torchao.dequantize_affine.default,
    )


def is_per_tensor_quant_node(node: fx.Node) -> bool:
    """Check if a quantize/dequantize node is per-tensor."""
    return node.target in (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    )


def is_per_channel_quant_node(node: fx.Node) -> bool:
    """Check if a quantize/dequantize node is per-channel."""
    return node.target in (
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    )


def get_qparams_from_node(
    node: fx.Node, prefix: str
) -> Optional[QuantParamsStruct[fx.Node]]:
    """Extract QuantParamsStruct from a node's flat args by prefix.

    Returns None if both scale and zero_point are None (unquantized).
    Raises ValueError if only one of scale/zero_point is None.
    """
    scale = get_arg(node, f"{prefix}_scale", Optional[fx.Node])
    zero_point = get_arg(node, f"{prefix}_zero_point", Optional[fx.Node])
    if scale is None and zero_point is None:
        return None
    if (scale is None) != (zero_point is None):
        raise ValueError(
            f"{prefix}_scale and {prefix}_zero_point must both be None or both be "
            f"provided on node {node.name}"
        )
    assert scale is not None and zero_point is not None
    return QuantParamsStruct(
        scale=scale,
        zero_point=zero_point,
        dtype=get_arg(node, f"{prefix}_dtype", torch.dtype),
        quant_min=get_arg(node, f"{prefix}_quant_min", int),
        quant_max=get_arg(node, f"{prefix}_quant_max", int),
    )


def get_qparams_flat(node: fx.Node, prefix: str) -> list[torch.fx.node.Argument]:
    """Get the 5 flat qparams values from a node for pass-through to another op."""
    return [
        get_arg(node, f"{prefix}_scale"),
        get_arg(node, f"{prefix}_zero_point"),
        get_arg(node, f"{prefix}_dtype"),
        get_arg(node, f"{prefix}_quant_min"),
        get_arg(node, f"{prefix}_quant_max"),
    ]


def get_scale(ep: ExportedProgram, qp: "QuantParamsStruct[fx.Node]") -> float:
    """Read a per-tensor scale value from its qparams.

    Per-tensor qparams are 0-dim lifted constants (placeholders) post-export;
    resolve the backing tensor and return its Python scalar.
    """
    value = get_constant(ep, qp.scale)
    assert value is not None, (
        f"expected a lifted-constant qparam for {qp.scale.name}, got op={qp.scale.op}"
    )
    return float(value.item())


def get_zero_point(ep: ExportedProgram, qp: "QuantParamsStruct[fx.Node]") -> int:
    """Read a per-tensor zero_point value from its qparams. See get_scale."""
    value = get_constant(ep, qp.zero_point)
    assert value is not None, (
        f"expected a lifted-constant qparam for {qp.zero_point.name}, "
        f"got op={qp.zero_point.op}"
    )
    return int(value.item())
