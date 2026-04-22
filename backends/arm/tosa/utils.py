# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Utility helpers for building TOSA graphs in the Arm backend."""

import logging
from typing import Any

import numpy as np

import torch
import tosa_serializer as ts

from executorch.backends.arm.tosa.mapping import extract_tensor_meta

from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node

logger = logging.getLogger(__name__)


def are_fake_tensors_broadcastable(
    fake_tensors: list[FakeTensor],
) -> tuple[bool, list[int]]:
    """Determine whether the fake tensors share a broadcastable shape.

    Args:
        fake_tensors (list[FakeTensor]): Fake tensors whose shapes should
            be validated for broadcasting.

    Returns:
        tuple[bool, list[int]]: Tuple where the first element indicates
            whether broadcasting is possible and the second element contains
            the broadcast shape. The shape list is empty when broadcasting
            fails.

    Raises:
        RuntimeError: Raised when fewer than two tensors are supplied.

    """
    if len(fake_tensors) < 1:
        raise RuntimeError(f"Expected 2 or more tensors got {len(fake_tensors)}")

    reversed_shapes = [list(reversed(ft.shape)) for ft in fake_tensors]
    sorted_shapes = sorted(reversed_shapes, key=len, reverse=True)

    broadcast_shape = []
    for dim in range(len(sorted_shapes[0])):
        curr_dim = 1
        for shape in sorted_shapes:
            if dim >= len(shape):
                continue
            if curr_dim == 1 and shape[dim] != 1:
                curr_dim = shape[dim]
            elif shape[dim] == 1:
                continue
            elif curr_dim != 1 and shape[dim] != curr_dim:
                return (False, [])
        broadcast_shape.append(curr_dim)
    return (True, list(reversed(broadcast_shape)))


def broadcast_tensors(tosa_fb, nodes: list[Node]) -> list[Any]:
    """Broadcast the FX nodes to a shared shape inside the TOSA graph.

    This mirrors ``reshape_for_broadcast`` but also emits the tile operators
    needed to materialize the broadcast and supports any number of inputs.

    Args:
        tosa_fb (Any): TOSA graph builder that receives the broadcast
            operators.
        nodes (list[Node]): FX nodes whose tensor metadata should be
            broadcast.
        tosa_spec (TosaSpecification): Active TOSA specification used to
            decode tensor metadata.

    Returns:
        list[Any]: Broadcast versions of the inputs. Each element is either
            the original FX node or a TOSA serializer tensor, ordered to match
            ``nodes``.

    Raises:
        RuntimeError: If the supplied nodes are not broadcastable.

    """
    index_fake_tensors = [node.meta["val"] for node in nodes]
    broadcastable, common_shape = are_fake_tensors_broadcastable(index_fake_tensors)
    if not broadcastable:
        raise RuntimeError("FakeTensors are not broadcastable")

    broadcast_tensors = []
    for node in nodes:
        tens_dtype, tens_shape = extract_tensor_meta(node.meta)
        list_tens_shape = list(tens_shape)
        # Already in the right shape we can just add it to the list.
        if list_tens_shape == common_shape:
            broadcast_tensors.append(node)
            continue

        rank_diff = len(common_shape) - len(tens_shape)
        new_shape = [1] * rank_diff + list_tens_shape
        reshaped = tosa_fb.addIntermediate(
            new_shape,
            tens_dtype,
        )

        build_reshape_tosa(tosa_fb, node.name, new_shape, reshaped.name)

        tiled = tosa_fb.addIntermediate(common_shape, tens_dtype)
        multipliers = [
            comm if curr == 1 else 1 for comm, curr in zip(common_shape, new_shape)
        ]
        multiple_shapes = tosa_fb.addConst(
            (len(multipliers),),
            ts.DType.SHAPE,
            multipliers,
            name=f"{node.name}_multiples",
        )

        attr = ts.TosaSerializerAttribute()
        attr.TileAttribute()
        tosa_fb.addOperator(
            ts.Op.TILE,
            [reshaped.name, multiple_shapes.name],
            [tiled.name],
            attr,
        )

        broadcast_tensors.append(tiled)

    return broadcast_tensors


def build_reshape_tosa(
    tosa_graph, input_name, new_shape, output_name, shape_name_override=""
):
    """Insert a TOSA reshape operator using the v1.0 semantics.

    Args:
        tosa_graph (Any): Graph builder used to emit TOSA operators.
        input_name (str): Name of the tensor that should be reshaped.
        new_shape (list[int]): Target tensor shape.
        output_name (str): Name assigned to the reshaped tensor.
        shape_name_override (str): Optional override for the shape constant
            name.

    """
    shape = tosa_graph.addConst(
        np.array(new_shape).shape,
        ts.DType.SHAPE,
        np.array(new_shape),
        name=shape_name_override if shape_name_override else output_name + "_shape",
    )

    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute()
    tosa_graph.addOperator(
        ts.Op.RESHAPE,
        [input_name, shape.name],
        [output_name],
        attr,
    )


def tosa_shape(shape):
    """Convert a shape tuple to a TOSA-compatible list, resolving symints."""
    return list([-1 if isinstance(d, torch.SymInt) else d for d in shape])
