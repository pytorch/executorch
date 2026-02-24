# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.accumulate_index_put_pass import (
    AccumulateIndexPutPass,
)
from executorch.backends.arm._passes.rewrite_index_put_pass import RewriteIndexPutPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_slice_scatter_ops = (exir_ops.edge.aten.slice_scatter.default,)
aten_slice_scatter_ops = (torch.ops.aten.slice_scatter.default,)


def _get_slice_scatter_decomposition(op) -> tuple:
    if op in edge_slice_scatter_ops:
        return (
            exir_ops.edge.aten.arange.start_step,
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.cat.default,
            exir_ops.edge.aten.permute_copy.default,
            exir_ops.edge.aten.index_put.default,
        )
    if op in aten_slice_scatter_ops:
        return (
            torch.ops.aten.arange.start_step,
            torch.ops.aten.slice_copy.Tensor,
            torch.ops.aten.cat.default,
            torch.ops.aten.permute_copy.default,
            torch.ops.aten.index_put.default,
        )
    raise RuntimeError(f"Can't get slice_scatter decomposition for op {op}")


def _fixup_start(start, dim_size: int) -> int:
    s = 0 if start is None else int(start)
    return max(0, min(s % dim_size if s < 0 else s, dim_size))


def _fixup_end(end, dim_size: int) -> int:
    e = dim_size if end is None else int(end)
    return max(0, min(e % dim_size if e < 0 else e, dim_size))


class DecomposeSliceScatterPass(ArmPass):
    """
    Decompose slice_scatter into:
      - Fast path (step == 1): slice_copy + cat (contiguous update), or
      - General path (step > 1): arange + index_put (strided / interleaved).

    Limitations:
      - Does not broadcast src: requires src.shape to exactly match the slice
        shape being updated

    For dim != 0, permute input/src so that the updated dimension is first,
    apply index_put with a single index tensor, then permute back.
    """

    _passes_required_after: Set[Type[ExportPass]] = {
        AccumulateIndexPutPass,
        RewriteIndexPutPass,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_slice_scatter_ops + aten_slice_scatter_ops):
            return super().call_operator(op, args, kwargs, meta)

        (
            arange_op,
            slice_copy_op,
            cat_op,
            permute_op,
            index_put_op,
        ) = _get_slice_scatter_decomposition(op)

        input = args[0]
        src = args[1]
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        start = args[3] if len(args) > 3 else kwargs.get("start", None)
        end = args[4] if len(args) > 4 else kwargs.get("end", None)
        step = args[5] if len(args) > 5 else kwargs.get("step", 1)

        if not isinstance(dim, int) or not isinstance(step, int):
            raise NotImplementedError("slice_scatter expects constant dim/step")

        if step <= 0:
            raise NotImplementedError("slice_scatter expects step > 0")

        input_val = input.data  # FakeTensor
        input_shape = input_val.shape  # [d0, d1, ..., d{r-1}]
        input_device = input_val.device
        input_rank = len(input_shape)
        dim_norm = dim % input_rank
        dim_size = int(input_shape[dim_norm])

        start_i = _fixup_start(start, dim_size)
        end_i = _fixup_end(end, dim_size)
        if end_i <= start_i:
            return input

        # index_positions: [W] where W = len(arange(start_i, end_i, step))
        index_positions = super().call_operator(
            arange_op,
            (start_i, end_i, step),
            {"dtype": torch.int32, "device": input_device},
            meta,
            updated=True,
        )

        src_val = src.data
        src_shape = src_val.shape
        index_shape = index_positions.data.shape
        # slice_shape is input_shape with dim_norm replaced by W
        # input_shape:      [d0, ..., D, ..., d{r-1}]
        #   -> slice_shape: [d0, ..., W, ..., d{r-1}]
        slice_shape = list(input_shape)
        slice_shape[dim_norm] = int(index_shape[0])
        # slice_scatter does not broadcast src: require exact shape match.
        if tuple(src_shape) != tuple(slice_shape):
            raise NotImplementedError(
                "slice_scatter requires src.shape to match the slice shape; "
                f"got src.shape={tuple(src_shape)}, expected={tuple(slice_shape)}"
            )

        # ---- fast path: contiguous update (step == 1) ----
        if step == 1:
            # prefix = input[..., :start_i, ...] along dim_norm
            prefix = super().call_operator(
                slice_copy_op,
                (input, dim_norm, 0, start_i, 1),
                {},
                meta,
                updated=True,
            )
            # suffix = input[..., end_i:, ...] along dim_norm
            suffix = super().call_operator(
                slice_copy_op,
                (input, dim_norm, end_i, dim_size, 1),
                {},
                meta,
                updated=True,
            )
            # concat(prefix, src, suffix) along dim_norm
            updated = super().call_operator(
                cat_op,
                ([prefix, src, suffix], dim_norm),
                {},
                meta,
                updated=True,
            )
            return updated

        # ---- general path: strided update (step > 1) ----
        # Move updated dim to front to use a single index tensor.
        if dim_norm != 0:
            perm = [dim_norm] + [i for i in range(input_rank) if i != dim_norm]
            inv_perm = [0] * input_rank
            for i, p in enumerate(perm):
                inv_perm[p] = i

            # input:           [d0, ..., d{dim_norm-1}, d{dim_norm}, d{dim_norm+1}, ..., d{r-1}]
            #   -> input_perm: [d{dim_norm}, d0, ..., d{dim_norm-1}, d{dim_norm+1}, ..., d{r-1}]
            input = super().call_operator(
                permute_op, (input, perm), {}, meta, updated=True
            )
            # src:           [d0, ..., d{dim_norm-1}, W, d{dim_norm+1}, ..., d{r-1}]
            #   -> src_perm: [W, d0, ..., d{dim_norm-1}, d{dim_norm+1}, ..., d{r-1}]
            src = super().call_operator(permute_op, (src, perm), {}, meta, updated=True)

        # Puts values from src into input along the first dimension
        # using a single 1D index tensor index_positions.
        updated = super().call_operator(
            index_put_op,
            (input, (index_positions,), src, False),
            {},
            meta,
            updated=True,
        )

        if dim_norm != 0:
            # updated_perm: [d{dim_norm}, ...] -> updated: [d0, d1, ..., d{r-1}]
            updated = super().call_operator(
                permute_op, (updated, inv_perm), {}, meta, updated=True
            )

        return updated
