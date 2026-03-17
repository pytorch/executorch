# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class RewriteHighRankSingletonPermutePass(ArmPass):
    """Rewrite high-rank permute via a lower-rank permute when singleton dims
    allow it.

    For rank>4 tensors, some backends are fragile around direct high-rank
    TRANSPOSE. When singleton dimensions are present, we can rewrite:

    permute(rank>4) -> view(remove singleton dims) -> permute(reduced rank) ->
    view(restore rank)

    This keeps semantics unchanged while reducing the permute rank.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _PERMUTE_OPS = (
        exir_ops.edge.aten.permute.default,
        exir_ops.edge.aten.permute_copy.default,
    )

    @staticmethod
    def _extract_permutation(permutation_arg: object) -> tuple[int, ...] | None:
        if not isinstance(permutation_arg, (list, tuple)):
            return None
        if not all(isinstance(dim, int) for dim in permutation_arg):
            return None
        return tuple(permutation_arg)

    @staticmethod
    def _normalize_permutation(
        permutation: Sequence[int], rank: int
    ) -> tuple[int, ...]:
        return tuple(dim % rank for dim in permutation)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._PERMUTE_OPS:
            return super().call_operator(op, args, kwargs, meta)
        if len(args) < 2:
            return super().call_operator(op, args, kwargs, meta)
        if not hasattr(args[0], "data"):
            return super().call_operator(op, args, kwargs, meta)
        if "val" not in meta or not hasattr(meta["val"], "shape"):
            return super().call_operator(op, args, kwargs, meta)

        permutation = self._extract_permutation(args[1])
        if permutation is None:
            return super().call_operator(op, args, kwargs, meta)

        input_shape = list(args[0].data.shape)
        output_shape = list(meta["val"].shape)
        rank = len(input_shape)
        if rank <= 4 or len(output_shape) != rank:
            return super().call_operator(op, args, kwargs, meta)

        normalized_permutation = self._normalize_permutation(permutation, rank)
        singleton_axes = [axis for axis, dim in enumerate(input_shape) if dim == 1]
        if not singleton_axes:
            return super().call_operator(op, args, kwargs, meta)

        non_singleton_axes = [
            axis for axis in range(rank) if axis not in singleton_axes
        ]
        reduced_rank = len(non_singleton_axes)
        if reduced_rank > 4:
            return super().call_operator(op, args, kwargs, meta)

        axis_to_reduced_axis = {
            axis: idx for idx, axis in enumerate(non_singleton_axes)
        }
        reduced_permutation = tuple(
            axis_to_reduced_axis[axis]
            for axis in normalized_permutation
            if axis in axis_to_reduced_axis
        )
        expected_axes = tuple(range(reduced_rank))
        if tuple(sorted(reduced_permutation)) != expected_axes:
            return super().call_operator(op, args, kwargs, meta)

        reduced_input_shape = [input_shape[axis] for axis in non_singleton_axes]
        reduced_input = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (args[0], reduced_input_shape),
            {},
            meta,
        )
        if reduced_permutation == expected_axes:
            reduced_output = reduced_input
        else:
            reduced_output = super().call_operator(
                op,
                (reduced_input, reduced_permutation),
                kwargs,
                meta,
            )
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (reduced_output, output_shape),
            {},
            meta,
        )
