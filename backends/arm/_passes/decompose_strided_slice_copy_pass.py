# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def _get_strided_slice_copy_decomposition(op):
    """Return the operator overloads used by this decomposition."""
    if op == exir_ops.edge.aten.slice_copy.Tensor:
        return (
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.view_copy.default,
        )
    raise RuntimeError(f"Can't get strided slice_copy decomposition for op {op}")


def _fixup_start(start, dim_size):
    """Normalize start and clamp into [0, dim_size]."""
    s = 0 if start is None else start
    if s < 0:
        s = s % dim_size
    return max(0, min(s, dim_size))


def _fixup_end(end, dim_size):
    """Normalize end and clamp into [0, dim_size]."""
    if end is None:
        return dim_size
    e = end
    if e > dim_size:
        e = dim_size
    if e < 0:
        e = e % dim_size
    return max(0, min(e, dim_size))


class DecomposeStridedSliceCopyPass(ArmPass):
    """
    Decompose edge.aten.slice_copy.Tensor with non-unit step into supported ops.

    Given:
        out = slice_copy(x, dim, start, end, step)   with step > 1

    Produce:
      1) y  = slice_copy(x, dim, start, end, 1)      # span with unit step
      2) pad y on the right to make length divisible by step (if needed)
      3) y2 = view_copy(y, ..., U, step, ...)        # split the sliced dim
      4) y3 = slice_copy(y2, dim_i + 1, 0, 1, 1)     # pick index 0 in each group
      5) out = view_copy(y3, ...)                    # collapse the singleton dim

    This implements "take every step-th element" using only unit-step slice + reshape.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    _TARGET_OPS = {exir_ops.edge.aten.slice_copy.Tensor}

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGET_OPS:
            return super().call_operator(op, args, kwargs, meta)

        # Only handle the non-unit-step case; leave unit-step to existing lowering.
        if not (len(args) == 5 and args[4] != 1):
            return super().call_operator(op, args, kwargs, meta)

        x, dim, start, end, step = args
        assert step > 0, "slice_copy step must be positive"

        shape = x.data.shape
        rank = len(shape)

        # Normalize dim into [0, rank).
        dim_i = dim % rank
        dim_size = shape[dim_i]

        # Normalize/clamp start/end into valid bounds.
        start_i = _fixup_start(start, dim_size)
        end_i = _fixup_end(end, dim_size)

        L = end_i - start_i
        if L <= 0:
            # slice_copy would return empty; keep default behavior.
            return super().call_operator(op, args, kwargs, meta)

        slice_op, cat_op, view_op = _get_strided_slice_copy_decomposition(op)

        # 1) Unit-step slice of the requested span:
        #    y = x[..., start_i:end_i, ...]
        y = super().call_operator(
            slice_op, (x, dim_i, start_i, end_i, 1), {}, meta, updated=True
        )

        # 2) Compute:
        #    U = ceil(L / step)  (# of output elements along dim_i)
        #    pad_right = U*step - L  (so that padded length becomes U*step)
        U = (L + step - 1) // step
        pad_right = U * step - L

        # 3) If needed, right-pad along dim_i so that:
        #    after padding, y.shape[dim_i] == U*step
        if pad_right > 0:
            y_data = y.data
            pad_shape = list(y_data.shape)
            pad_shape[dim_i] = pad_right

            # z: zeros with same dtype/device as y, shape matches y except
            # z.shape[dim_i] = pad_right.
            fill_value = False if y_data.dtype == torch.bool else 0
            z = super().call_operator(
                op=exir_ops.edge.aten.full.default,
                args=(pad_shape, fill_value),
                kwargs={"dtype": y_data.dtype, "device": y_data.device},
                meta=meta,
                updated=True,
            )

            # Concatenate on the right:
            #   y.shape[dim_i] : L -> L + pad_right == U*step
            y = super().call_operator(cat_op, ([y, z], dim_i), {}, meta, updated=True)

        # 4) Split the sliced dim: (U*step) -> (U, step)
        y_t2 = y.data
        split_shape = list(y_t2.shape)
        split_shape[dim_i] = U
        split_shape.insert(dim_i + 1, step)

        y2 = super().call_operator(view_op, (y, split_shape), {}, meta, updated=True)

        # 5) Take index 0 in the inserted "step" dimension:
        #    [..., U, step, ...] -> [..., U, 1, ...]
        y3 = super().call_operator(
            slice_op, (y2, dim_i + 1, 0, 1, 1), {}, meta, updated=True
        )

        # 6) Collapse y3's singleton step dim: [..., U, 1, ...] -> [..., U, ...].
        out_shape = list(y_t2.shape)  # y_t2: [..., U*step, ...]
        out_shape[dim_i] = U  # out_shape: [..., U, ...]

        return super().call_operator(view_op, (y3, out_shape), {}, meta, updated=True)
