# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Utility helpers for building TOSA graphs in the Arm backend."""

import logging
from typing import Any

import numpy as np

import sympy  # type: ignore

import torch
import tosa_serializer as ts

from executorch.backends.arm.tosa.mapping import extract_tensor_meta
from executorch.backends.arm.tosa.specification import TosaSpecification

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


def broadcast_tensors(
    tosa_fb, nodes: list[Node], tosa_spec: TosaSpecification
) -> list[Any]:
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


def tosa_shape(shape, dim_order):
    """Reorder a shape tuple into TOSA layout while resolving symints.

    Args:
        shape (Sequence[int | torch.SymInt]): Original tensor shape,
            possibly containing ``torch.SymInt``.
        dim_order (Sequence[int]): Desired dimension order for the output
            shape.

    Returns:
        list[int]: List containing the reordered dimensions where symbolic
            values become ``-1``.

    """
    reordered = tuple([shape[dim] for dim in dim_order])
    # Dynamic shapes in executorch are represented with torch.SymInt objects in the shapes,
    # in TOSA we do not have this concept and instead use -1.
    removed_symints = tuple(
        [-1 if isinstance(d, torch.SymInt) else d for d in reordered]
    )
    return list(removed_symints)


def get_resize_parameters_1d(
    input_size: int | torch.SymInt,
    output_size: int | torch.SymInt,
    resize_mode: int,
    align_corners: bool,
):
    """Compute resize coefficients for a single spatial dimension.

    Args:
        input_size (int | torch.SymInt): Input size for the axis, possibly
            symbolic.
        output_size (int | torch.SymInt): Output size for the axis, possibly
            symbolic.
        resize_mode (int): Target resize mode defined by TOSA.
        align_corners (bool): Whether the resize should align the corner
            pixels.

    Returns:
        tuple[int, int, int, int]: Numerator, denominator, offset, and border
            terms encoded as integers.

    Raises:
        RuntimeError: If symbolic shapes are used with ``align_corners`` or if
            the computed ratio or border is not constant.

    """
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
    """Calculate 2D resize parameters for TOSA emission.

    Args:
        input_size_xy (tuple[int | torch.SymInt, int | torch.SymInt]): Height
            and width of the input tensor.
        output_size_xy (tuple[int | torch.SymInt, int | torch.SymInt]): Height
            and width of the output tensor.
        resize_mode (int): TOSA resize mode used for coefficient generation.
        align_corners (bool): Whether to align corner pixels between input and
            output.

    Returns:
        tuple[torch.IntTensor, ...]: Four-element tuple of tensors describing
            the scale numerator, scale denominator, offset, and border for Y
            and X dimensions.

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
