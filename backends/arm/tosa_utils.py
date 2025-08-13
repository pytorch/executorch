# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from typing import Any

import numpy as np
import serializer.tosa_serializer as ts  # type: ignore

import sympy  # type: ignore

import torch

from executorch.backends.arm.tosa_mapping import extract_tensor_meta

from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops

from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node

logger = logging.getLogger(__name__)


def are_fake_tensors_broadcastable(
    fake_tensors: list[FakeTensor],
) -> tuple[bool, list[int]]:
    """
    Determines whether a list of FakeTensors can be broadcast together.
    Args:
        fake_tensors (list[FakeTensor]): List of 2 or more FakeTensors
        who's shapes to evaluate

    Returns:
        tuple[bool, list[int]]: First element is whether the shapes are
        broadcastable. Second element is the common shape if compatible.
        If not, empty list.

    Raises:
        RuntimeError: If less than 2 tensors are passed in.
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


def broadcast_tensors(
    tosa_fb, nodes: list[Node], tosa_spec: TosaSpecification
) -> list[Any]:
    """
    Given a list of nodes it determines the common shape they broadcast to
    and adds the necessary reshape and tile operations to perform the broadcast.

    Args:
        tosa_fb: Tosa graph to add nodes to
        nodes (list[Node]): List of nodes to broadcast together
        tosa_spec (TosaSpecification): Tosa spec

    Returns:
        list[Any]: List containing the fx.Nodes or TosaSerializerTensors
        of the right common shape. Order of output matches order of input.

    Raises:
        RuntimeError: If the supplied nodes are not broadcastable.

    Note:
        This function and `reshape_for_broadcast` both reshape the tensors
        for broadcast. However this function also performs the broadcast and
        does not have a limit on only two input tensors.
    """
    index_fake_tensors = [node.meta["val"] for node in nodes]
    broadcastable, common_shape = are_fake_tensors_broadcastable(index_fake_tensors)
    if not broadcastable:
        raise RuntimeError("FakeTensors are not broadcastable")

    broadcast_tensors = []
    for node in nodes:
        tens_dtype, tens_shape, _ = extract_tensor_meta(node.meta, tosa_spec)
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

        build_reshape_tosa_1_0(tosa_fb, node.name, new_shape, reshaped.name)

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

        tosa_fb.addOperator(
            ts.TosaOp.Op().TILE,
            [reshaped.name, multiple_shapes.name],
            [tiled.name],
            None,
        )

        broadcast_tensors.append(tiled)

    return broadcast_tensors


def build_reshape_tosa_1_0(
    tosa_graph, input_name, new_shape, output_name, shape_name_override=""
):
    shape = tosa_graph.addConst(
        np.array(new_shape).shape,
        ts.DType.SHAPE,
        np.array(new_shape),
        name=shape_name_override if shape_name_override else output_name + "_shape",
    )

    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute()
    tosa_graph.addOperator(
        ts.TosaOp.Op().RESHAPE,
        [input_name, shape.name],
        [output_name],
        attr,
    )


def is_consumer_node_depthwise_conv2d(node: Node):
    consumer_node = list(node.users)[0]
    if consumer_node.target == exir_ops.edge.aten.convolution.default:
        consumer_node_inputs = consumer_node.all_input_nodes
        groups = consumer_node.args[-1]
        in_channels = consumer_node_inputs[0].meta["val"].shape[1]
        out_channels = consumer_node_inputs[1].meta["val"].shape[0]
        if (in_channels == groups) and (out_channels % in_channels) == 0:
            return True

    return False


def tosa_shape(shape, dim_order):
    reordered = tuple([shape[dim] for dim in dim_order])
    # Dynamic shapes in executorch are represented with torch.SymInt objects in the shapes,
    # in TOSA we do not have this concept and instead use -1.
    removed_symints = tuple(
        [-1 if isinstance(d, torch.SymInt) else d for d in reordered]
    )
    return removed_symints


def get_resize_parameters_1d(
    input_size: int | torch.SymInt,
    output_size: int | torch.SymInt,
    resize_mode: int,
    align_corners: bool,
):
    # We don't support align_corners for symbolic shapes, because handling the edge case where size == 1 is tricky.
    if align_corners:
        if (not isinstance(input_size, int)) or (not isinstance(output_size, int)):
            raise RuntimeError(
                "We do not support align_corners=True for symbolic shapes."
            )

    # SymInt seems to not actually work for symbolic expressions, so use the underlying sympy objects instead
    input_size = (
        input_size.node._expr if isinstance(input_size, torch.SymInt) else input_size
    )
    output_size = (
        output_size.node._expr if isinstance(output_size, torch.SymInt) else output_size
    )
    if align_corners and input_size > 1 and output_size > 1:
        scale_n = output_size - 1
    else:
        scale_n = output_size
    if align_corners and input_size > 1 and output_size > 1:
        scale_d = input_size - 1
    else:
        scale_d = input_size
    ratio = scale_n / scale_d
    if not sympy.sympify(ratio).is_constant():
        raise RuntimeError(
            "Resize requires a constant ratio: " + str(ratio) + " is not constant!"
        )
    gcd = sympy.gcd(scale_n, scale_d)
    scale_n = 2 * scale_n // gcd
    scale_d = 2 * scale_d // gcd
    # These should always be whole integers, based on the above calculations
    scale_n = int(scale_n.evalf())
    scale_d = int(scale_d.evalf())

    if align_corners:
        offset = 0
    else:
        # Half pixel centers so input and output sampling positions are offset by 1/2 pixel.
        offset = scale_d // 2 - scale_n // 2

    # Calculate border to maintain the correct the output size.
    # Note that this should always result in a constant value, as the ratio is constant.
    border = scale_d * (output_size - 1) - scale_n * (input_size - 1) + offset

    if not sympy.sympify(border).is_constant():
        raise RuntimeError(
            "Resize requires a constant border: " + str(border) + " is not constant!"
        )

    border = int(sympy.sympify(border).evalf())
    return scale_n, scale_d, offset, border


def get_resize_parameters(
    input_size_xy: tuple[int | torch.SymInt, int | torch.SymInt],
    output_size_xy: tuple[int | torch.SymInt, int | torch.SymInt],
    resize_mode: int,
    align_corners: bool,
) -> tuple[torch.IntTensor, ...]:
    """Get the tosa.resize parameters based on the input and output size.

    Args:
        input_size_xy (tuple[int | torch.SymInt]): Size of the input
        output_size_xy (tuple[int | torch.SymInt]): Size of the output
        resize_mode (tosa.ResizeMode): The TOSA resize mode
        align_corners (bool): Align the corners pixels of the input and output

    Returns:
        scale_n (torch.IntTensor), scale_d (torch.IntTensor),
        offset (torch.IntTensor), border (torch.IntTensor)
    """

    # Get the parameters for each dimension independently
    y_params = get_resize_parameters_1d(
        input_size_xy[0], output_size_xy[0], resize_mode, align_corners
    )
    x_params = get_resize_parameters_1d(
        input_size_xy[1], output_size_xy[1], resize_mode, align_corners
    )
    # Combine them together, so we return four 2-element tensors (scale_n, scale_d, offset, border)
    return tuple(map(torch.IntTensor, zip(y_params, x_params)))
