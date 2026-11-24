# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Build TOSA operations with compact compile-time parameters."""

import math
from typing import cast

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops.conversion_ops import (
    validate_rescale_zero_points,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph, Node


def compute_multiplier_and_shift(scales: list[float], scale_width: int = 32):
    if scale_width not in (16, 32):
        raise ValueError(f"Unsupported scale width: {scale_width}")
    multipliers, shifts = [], []
    offset = scale_width - 1
    for scale in scales:
        mantissa, exponent = math.frexp(scale)
        multiplier = round(mantissa * (1 << offset))
        if multiplier == 1 << offset:
            multiplier //= 2
            exponent += 1
        shift = offset - exponent
        if shift > 62:
            multiplier >>= min(31, shift - 62)
            shift = 62
        if multiplier < 0 or not 2 <= shift <= 62:
            raise ValueError(f"Scale {scale} cannot be represented by TOSA RESCALE")
        multipliers.append(multiplier)
        shifts.append(shift)
    return multipliers, shifts


def create_const(graph: Graph, values, dtype: torch.dtype, from_node: Node) -> Node:
    if isinstance(values, torch.Tensor):
        values = values.flatten().tolist()
    elif not isinstance(values, (tuple, list)):
        values = [values]
    values = list(values)
    node = create_node(
        graph,
        exir_ops.backend.tosa.CONST.default,
        (values, [len(values)], dtype),
        from_node=from_node,
    )
    node.meta.pop(TosaSpecialDtype.meta_key(), None)
    fake = from_node.meta["val"]
    node.meta["val"] = fake.new_empty((len(values),), dtype=dtype)
    return node


def set_rescale_args(
    node: Node,
    input_node: Node,
    dtype: torch.dtype,
    scales: list[float],
    input_zp: int,
    output_zp: int,
    *,
    input_unsigned: bool = False,
    output_unsigned: bool = False,
) -> None:
    graph = node.graph
    input_dtype = cast(torch.dtype, input_node.meta["val"].dtype)
    validate_rescale_zero_points(
        input_dtype, dtype, input_zp, output_zp, input_unsigned, output_unsigned
    )
    if output_unsigned and input_dtype not in (torch.int8, torch.int16, torch.uint8):
        if dtype != torch.int8:
            raise ValueError("Wide-to-unsigned RESCALE only supports int8 output")
        with graph.inserting_before(node):
            signed = create_rescale_node(
                graph,
                (input_node, dtype, scales, input_zp, output_zp - 128),
                from_node=node,
            )
        input_node, scales, input_zp, output_zp = signed, [1.0], -128, 0
    scale32 = input_node.meta.get(TosaSpecialDtype.meta_key()) != TosaSpecialDtype.INT48
    multipliers, shifts = compute_multiplier_and_shift(scales, 32 if scale32 else 16)
    node.args = (input_node, multipliers, shifts, input_zp, output_zp)
    node.kwargs = {
        "dtype": dtype,
        "scale32": scale32,
        "rounding_mode": "SINGLE_ROUND",
        "per_channel": len(scales) > 1,
        "input_unsigned": input_unsigned,
        "output_unsigned": output_unsigned,
    }
    node.meta.pop(TosaSpecialDtype.meta_key(), None)
    node.meta["val"] = input_node.meta["val"].new_empty(
        input_node.meta["val"].shape, dtype=dtype
    )


def create_rescale_node(
    graph: Graph,
    args: tuple = (),
    kwargs: dict | None = None,
    from_node: Node | None = None,
) -> Node:
    node = create_node(
        graph, exir_ops.backend.tosa.RESCALE.default, from_node=from_node
    )
    if args:
        input_node = args[0] if args[0] is not None else from_node
        assert isinstance(input_node, Node)
        set_rescale_args(node, input_node, *args[1:], **(kwargs or {}))
        if args[0] is None:
            node.update_arg(0, None)
    return node


def rescale_parameters(node: Node) -> tuple[list[float], int, int, torch.dtype]:
    multipliers, shifts, input_zp, output_zp = cast(
        tuple[list[int], list[int], int, int], node.args[1:]
    )
    scales = [
        multiplier / (1 << shift)
        for multiplier, shift in zip(multipliers, shifts, strict=True)
    ]
    return scales, input_zp, output_zp, cast(torch.dtype, node.kwargs["dtype"])


def rescale_tensor(
    input_tensor: torch.Tensor,
    dtype: torch.dtype,
    scales: list[float],
    input_zp: int,
    output_zp: int,
    *,
    input_unsigned: bool = False,
    output_unsigned: bool = False,
) -> torch.Tensor:
    validate_rescale_zero_points(
        input_tensor.dtype, dtype, input_zp, output_zp, input_unsigned, output_unsigned
    )
    if output_unsigned and input_tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.uint8,
    ):
        if dtype != torch.int8:
            raise TosaValueError("Wide-to-unsigned RESCALE only supports int8 output")
        signed = rescale_tensor(input_tensor, dtype, scales, input_zp, output_zp - 128)
        return rescale_tensor(signed, dtype, [1.0], -128, 0, output_unsigned=True)
    multipliers, shifts = compute_multiplier_and_shift(scales)
    return exir_ops.backend.tosa.RESCALE.default(
        input_tensor,
        multipliers,
        shifts,
        input_zp,
        output_zp,
        dtype=dtype,
        per_channel=len(scales) > 1,
        input_unsigned=input_unsigned,
        output_unsigned=output_unsigned,
    )
