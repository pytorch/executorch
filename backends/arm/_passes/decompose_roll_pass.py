# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


RollParameters = tuple[tuple[int, int, int], ...]


def get_static_roll_parameters(
    input_shape: Sequence[object], shifts: object, dims: object
) -> RollParameters | None:
    """Normalize a statically-decomposable roll.

    Args:
        input_shape (Sequence[object]): Shape of the roll input.
        shifts (object): Roll shifts from the operator arguments.
        dims (object): Roll dimensions from the operator arguments.

    Returns:
        RollParameters | None: Normalized ``(shift, dim, size)`` tuples, or
        ``None`` when the roll cannot be decomposed statically.

    """
    if not input_shape or any(
        type(size) is not int or size <= 0 for size in input_shape
    ):
        return None
    if not isinstance(shifts, (list, tuple)) or not isinstance(dims, (list, tuple)):
        return None
    if not shifts or len(shifts) != len(dims):
        return None
    if any(type(value) is not int for value in (*shifts, *dims)):
        return None

    rank = len(input_shape)
    parameters: list[tuple[int, int, int]] = []
    for shift, dim in zip(shifts, dims):
        if not -rank <= dim < rank:
            return None
        normalized_dim = dim % rank
        dim_size = int(input_shape[normalized_dim])
        parameters.append((shift % dim_size, normalized_dim, dim_size))
    return tuple(parameters)


def can_decompose_roll(
    input_shape: Sequence[object], shifts: object, dims: object
) -> bool:
    """Return whether a roll can become a nonempty slice/concat graph.

    Args:
        input_shape (Sequence[object]): Shape of the roll input.
        shifts (object): Roll shifts from the operator arguments.
        dims (object): Roll dimensions from the operator arguments.

    Returns:
        bool: True when the roll has a supported static decomposition.

    """
    parameters = get_static_roll_parameters(input_shape, shifts, dims)
    return parameters is not None and any(shift != 0 for shift, _, _ in parameters)


class DecomposeRollPass(ArmOpTargetedPass):
    """Decompose a static ``aten.roll`` into slices and concatenation.

    For each nonzero ``(shift, dim)`` pair, normalize the shift and apply:

        shift = shift % dim_size
        result = cat(
            (
                slice_copy(result, dim, dim_size - shift, dim_size),
                slice_copy(result, dim, 0, dim_size - shift),
            ),
            dim,
        )

    Rewrites are applied sequentially to support multiple and repeated
    dimensions.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = {exir_ops.edge.aten.roll.default}

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        input_tensor = args[0]
        shifts = args[1]
        dims = args[2] if len(args) > 2 else ()
        parameters = get_static_roll_parameters(input_tensor.data.shape, shifts, dims)
        if parameters is None or not any(shift != 0 for shift, _, _ in parameters):
            raise ValueError("Expected a nonempty static roll decomposition")

        result = input_tensor
        for shift, dim, dim_size in parameters:
            if shift == 0:
                continue
            split = dim_size - shift
            suffix = super().call_operator(
                exir_ops.edge.aten.slice_copy.Tensor,
                (result, dim, split, dim_size, 1),
                {},
                meta,
                updated=True,
            )
            prefix = super().call_operator(
                exir_ops.edge.aten.slice_copy.Tensor,
                (result, dim, 0, split, 1),
                {},
                meta,
                updated=True,
            )
            result = super().call_operator(
                exir_ops.edge.aten.cat.default,
                ([suffix, prefix], dim),
                {},
                meta,
                updated=True,
            )
        return result
