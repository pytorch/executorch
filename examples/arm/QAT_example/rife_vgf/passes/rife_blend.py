# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
from executorch.backends.arm.operators.op_tosa_rescale import (
    _compute_multiplier_and_shift,
)
from torch.fx import Node

from .pack_rife_warps_and_concat import (
    _constant_tensor,
    _is_op,
    _shape_values,
    _tensor_shape,
    TOSA,
)


@dataclass
class RifeBlend:
    output: Node
    rescales: tuple[Node, Node, Node, Node, Node]
    complement: torch.Tensor
    image_zero_points: tuple[int, int]
    mask_zero_points: tuple[int, int]
    output_rescale: Node | None = None


def _rescale(node, dtype, nodes, *, output_unsigned=False):
    output_min, output_max = (0, 255) if output_unsigned else (-128, 127)
    if (
        not _is_op(node, TOSA.RESCALE.default, 5)
        or node.args[1] != dtype
        or len(node.args[2]) != 1
        or not math.isfinite(node.args[2][0])
        or node.args[2][0] <= 0
        or node.kwargs.get("output_unsigned", False) != output_unsigned
        or node.kwargs.get("input_unsigned", False)
        or not all(type(value) is int for value in node.args[3:5])
        or not -128 <= node.args[3] <= 127
        or not output_min <= node.args[4] <= output_max
        or (output_unsigned and node.args[0].meta["val"].dtype != torch.int8)
        or (dtype == torch.int32 and node.args[4] != 0)
        or (node.args[0].meta["val"].dtype == torch.int32 and node.args[3] != 0)
    ):
        raise ValueError(
            "Unsupported blend rescale dtype, scale, zero points or flags."
        )
    nodes.add(node)
    return node.args[0]


def _apply_rescale(values, node, *, clamp=True):
    multipliers, shifts = _compute_multiplier_and_shift(node.args[2])
    scaled = (
        (values.to(torch.int64) - node.args[3]) * multipliers[0]
        + (1 << (shifts[0] - 1))
    ) >> shifts[0]
    if not clamp:
        return scaled + node.args[4]
    limits = torch.iinfo(
        torch.uint8 if node.kwargs.get("output_unsigned", False) else node.args[1]
    )
    return (scaled + node.args[4]).clamp(limits.min, limits.max)


def _weighted_image(branch, warp, shape, nodes):
    multiply_rescale = _rescale(branch, torch.int32, nodes)
    multiply = _rescale(multiply_rescale, torch.int8, nodes)
    if not _is_op(multiply, TOSA.MUL.default, 3):
        raise ValueError("Expected an image-mask product with a tensor shift.")
    shift = multiply.args[2]
    if not _is_op(shift, TOSA.CONST.default, 3) or shift.args != ([0], [1], torch.int8):
        raise ValueError("Expected an unshifted image-mask product.")
    nodes.add(multiply)
    image_rescale, mask_rescale = multiply.args[:2]
    crop = _rescale(image_rescale, torch.int32, nodes)
    mask = _rescale(mask_rescale, torch.int32, nodes)
    if any(n.args[2] != [1.0] for n in (image_rescale, mask_rescale)):
        raise ValueError("Expected unit-scale image and mask centering.")
    if (
        not _is_op(crop, TOSA.SLICE.default, 3)
        or _shape_values(crop.args[1]) != [0, 0, 0, 0]
        or _shape_values(crop.args[2]) != list(shape)
    ):
        raise ValueError("Expected an RGB warp crop.")
    transpose = crop.args[0]
    if (
        not _is_op(transpose, TOSA.TRANSPOSE.default, 2)
        or _shape_values(transpose.args[1]) != [0, 3, 1, 2]
        or transpose.args[0] is not warp
    ):
        raise ValueError("Expected the original warped image layout.")
    nodes.update((crop, transpose))
    return multiply_rescale, mask, image_rescale.args[3], mask_rescale.args[3]


def _complement_table(program, complement, sigmoid, nodes):
    subtract = _rescale(complement, torch.int8, nodes)
    if not _is_op(subtract, TOSA.SUB.default, 2):
        raise ValueError("Expected the mask complement subtraction.")
    nodes.add(subtract)
    constant_rescale, mask_rescale = subtract.args
    constant = _rescale(constant_rescale, torch.int32, nodes)
    if _rescale(mask_rescale, torch.int32, nodes) is not sigmoid:
        raise ValueError("Expected the shared sigmoid mask.")
    value = _constant_tensor(program, constant)
    if value.dtype != torch.int8 or value.numel() != 1:
        raise ValueError("Expected a scalar INT8 blend constant.")
    table = _constant_tensor(program, sigmoid.args[1])
    difference = _apply_rescale(value.reshape(()), constant_rescale) - _apply_rescale(
        table, mask_rescale
    )
    if torch.any(difference < -(1 << 31)) or torch.any(difference >= (1 << 31)):
        raise ValueError("Mask complement could overflow INT32.")
    return _apply_rescale(difference, complement).to(torch.int8)


def match_rife_blend(program, warps, sigmoid):
    """Match a private quantized RGB blend and preserve its rescale boundaries.

    Args:
        program (ExportedProgram): Program containing immutable blend constants.
        warps (list[Node]): Original two NHWC warp outputs.
        sigmoid (Node): Signed INT8 mask lookup.

    Returns:
        RifeBlend: Blend output, integer rescales and exact complement lookup.

    """
    graph = sigmoid.graph
    output = graph.output_node()
    if len(output.args[0]) != 1:
        raise ValueError("Expected one final image output.")
    result = output.args[0][0]
    _, height, width, _ = _tensor_shape(warps[0])
    shape = (1, 3, height, width)
    if _tensor_shape(result) != shape or result.meta["val"].dtype != torch.int8:
        raise ValueError("Expected an NCHW INT8 RGB result.")
    nodes = {*warps, sigmoid}
    output_rescale = None
    blend_result = result
    if result.kwargs.get("output_unsigned", False):
        blend_result = _rescale(result, torch.int8, nodes, output_unsigned=True)
        if _tensor_shape(blend_result) != shape:
            raise ValueError("Expected an NCHW INT8 blend before the output rescale.")
        output_rescale = result
    add = _rescale(blend_result, torch.int8, nodes)
    if not _is_op(add, TOSA.ADD.default, 2):
        raise ValueError("Expected the final weighted image addition.")
    nodes.add(add)
    left, right = add.args
    mul0, mask0, image_zp0, mask_zp0 = _weighted_image(left, warps[0], shape, nodes)
    mul1, mask1, image_zp1, mask_zp1 = _weighted_image(right, warps[1], shape, nodes)
    if mask0 is not sigmoid:
        raise ValueError("Expected sigmoid weighting on the first image.")
    complement = _complement_table(program, mask1, sigmoid, nodes)
    if any(user not in nodes | {output} for node in nodes for user in node.users):
        raise ValueError("The image blend has additional consumers.")
    # The shader's INT32 add must not depend on TOSA saturation or overflow.
    codes = torch.tensor([-128, 127], dtype=torch.int64)
    left_bounds = _apply_rescale(codes, left, clamp=False)
    right_bounds = _apply_rescale(codes, right, clamp=False)
    limits = torch.cat((left_bounds, right_bounds, left_bounds + right_bounds))
    if torch.any(limits < -(1 << 31)) or torch.any(limits >= (1 << 31)):
        raise ValueError("Weighted image addition could overflow INT32.")
    return RifeBlend(
        result,
        (mul0, mul1, left, right, blend_result),
        complement,
        (image_zp0, image_zp1),
        (mask_zp0, mask_zp1),
        output_rescale,
    )
