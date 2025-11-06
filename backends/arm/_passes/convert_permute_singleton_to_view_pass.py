# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Sequence, Set, Tuple, Type

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

from torch._ops import OpOverload


_PERMUTE_TARGETS: Tuple[OpOverload, ...] = (
    exir_ops.edge.aten.permute.default,
    exir_ops.edge.aten.permute_copy.default,
)


class ConvertPermuteSingletonToViewPass(ExportPass):
    """Replace permutations that only move singleton axes with a reshape.

    Examples:
    x = rand(1,1,1,4)
    y = permute(x, (0,3,1,2))

    becomes:
    x = rand(1,1,1,4)
    y = view_copy(x, (1,4,1,1))
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in _PERMUTE_TARGETS:
            return super().call_operator(op, args, kwargs, meta)

        input_tensor = args[0].data
        permutation = args[1]
        if not is_singleton_permutation(input_tensor.shape, permutation):
            return super().call_operator(op, args, kwargs, meta)

        output_shape = meta["val"].shape
        view_args = (args[0], output_shape)
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default, view_args, kwargs, meta
        )


def is_singleton_permutation(shape: Sequence[int], permutation: Sequence[int]) -> bool:
    """
    Treat as a view only when non-singleton axes keep their order; singleton
    axes may move freely since they carry no data volume.
    """
    rank = len(shape)
    normalized_perm = [d % rank for d in permutation]

    non_singleton_axes = [i for i, size in enumerate(shape) if size != 1]
    permuted_non_singleton_axes = [axis for axis in normalized_perm if shape[axis] != 1]

    return permuted_non_singleton_axes == non_singleton_axes
