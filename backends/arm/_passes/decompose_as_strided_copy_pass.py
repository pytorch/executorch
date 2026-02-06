# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Set, Tuple, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.common.as_strided_utils import (
    contiguous_strides,
    maybe_static_sequence,
    to_int,
    to_int_tuple,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeAsStridedCopyPass(ArmPass):
    """
    Replace contiguous `aten.as_strided_copy` with `aten.view_copy`.

    The TOSA backend only supports the contiguous-as-strided case where the stride matches
    row-major layout and the storage offset is zero. In that scenario the operator is
    equivalent to a reshape with copy semantics and can be lowered via `view_copy`.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _EDGE_OPS = (exir_ops.edge.aten.as_strided_copy.default,)
    _ATEN_OPS = (torch.ops.aten.as_strided_copy.default,)

    def _extract_args(
        self, args: Tuple[object, ...], kwargs: dict
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...], int]]:
        """Return (size, stride, storage_offset) when they are statically known."""
        if len(args) < 3:
            return None

        size_arg = args[1]
        stride_arg = args[2]
        offset_arg = (
            kwargs.get("storage_offset") if "storage_offset" in kwargs else None
        )
        if offset_arg is None and len(args) > 3:
            offset_arg = args[3]

        size_seq = maybe_static_sequence(size_arg)
        stride_seq = maybe_static_sequence(stride_arg)
        if size_seq is None or stride_seq is None:
            return None

        size_tuple = to_int_tuple(size_seq)
        stride_tuple = to_int_tuple(stride_seq)
        if size_tuple is None or stride_tuple is None:
            return None

        if len(size_tuple) != len(stride_tuple):
            return None

        if any(stride < 0 for stride in stride_tuple):
            return None

        if offset_arg is None:
            storage_offset = 0
        else:
            parsed_offset = to_int(offset_arg)
            if parsed_offset is None:
                return None
            storage_offset = parsed_offset

        return size_tuple, stride_tuple, storage_offset

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
        if op not in (*self._EDGE_OPS, *self._ATEN_OPS):
            return super().call_operator(op, args, kwargs, meta, updated)

        extracted = self._extract_args(args, kwargs)
        if extracted is None:
            return super().call_operator(op, args, kwargs, meta, updated)

        size_tuple, stride_tuple, storage_offset = extracted
        if storage_offset != 0:
            return super().call_operator(op, args, kwargs, meta, updated)

        expected_strides = contiguous_strides(size_tuple)

        def _stride_matches(idx: int, dim: int) -> bool:
            stride = stride_tuple[idx]
            expected = expected_strides[idx]
            if idx == len(size_tuple) - 1:
                return stride >= expected
            if dim == 1 or expected == 0:
                return True
            return stride == expected

        if any(not _stride_matches(i, dim) for i, dim in enumerate(size_tuple)):
            return super().call_operator(op, args, kwargs, meta, updated)

        view_args = (args[0], tuple(size_tuple))
        view_kwargs: Dict[str, object] = {}

        view_op = (
            exir_ops.edge.aten.view_copy.default
            if op in self._EDGE_OPS
            else torch.ops.aten.view_copy.default
        )

        return super().call_operator(
            view_op, view_args, view_kwargs, meta, updated=True
        )
