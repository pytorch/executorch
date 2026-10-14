# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from math import prod
from typing import Optional

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class MoveSliceBeforeViewPass(RemoveOrReplacePassInterface):
    """Move a slice_copy above a view_copy when the slice is re-expressible as a
    single slice on one dim of the pre-view tensor.

    Rewrites  view(x) -> slice(dim=d, start, end, step)  into
    slice(x, dim=d', start', end', step') -> view(sliced, slice_out_shape), so the
    slice lands directly on x. This may be useful in attention patterns, where
    we view outputs of a large linear into a new shape where the number of
    attention heads are the last dim, and we need to run independent computation
    per head. Moving the slice before the view can allow us to then directly slice
    the constant linear weights.

    A view is a contiguous reshape: it never moves or reorders elements, it only
    re-groups the shared row-major index space into different dims. A slice keeps
    an arithmetic progression of indices (start, start+step, ...) along one viewed
    dim, and that progression collapses back to a *single* slice on one pre-view
    dim exactly when the row-major strides line up. ``_derive_pre_view_slice``
    handles the three cases that qualify:

      * untouched dim: the viewed dim is left unchanged by the view -- same size
        and same inner stride as some pre-view dim -- so the slice copies over
        verbatim (any step).
      * contiguous: the viewed dim and a pre-view dim span the same flat extent
        (a split's outermost factor, or a merge that aligns), so a contiguous
        (step==1) slice maps to a contiguous pre-view slice.
      * strided: the viewed dim is an innermost factor of a pre-view dim
        (identical inner stride) selected width-1, so it maps to a strided
        pre-view slice with step == the viewed dim's size.

    Everything else -- middle factors, wider strided selections -- is block-strided
    (runs separated by gaps), which no single slice can express, so it is left
    unchanged.

    Each slice is handled independently, so a view that fans out to several slices
    is rewritten one slice at a time and the now-dead view is removed by dead-code
    elimination -- there is no single-user requirement on the view.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.slice_copy.Tensor]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        view_node = get_arg(node, "input", torch.fx.Node)
        if view_node.target != exir_ops.edge.aten.view_copy.default:
            return False

        x_node = get_arg(view_node, "input", torch.fx.Node)
        pre_view_shape = tuple(x_node.meta["val"].shape)
        post_view_shape = tuple(view_node.meta["val"].shape)
        if 0 in pre_view_shape or 0 in post_view_shape:
            return False

        dim = get_arg(node, "dim", int)
        if dim < 0:
            dim += len(post_view_shape)
        post_view_size = post_view_shape[dim]

        bounds = self._normalize_slice(node, post_view_size)
        if bounds is None:
            return False
        start, stop, step = bounds

        # The slice's own output shape gives the selected-element count along the
        # sliced dim directly -- it is exactly output_shape[dim].
        slice_out_shape = tuple(node.meta["val"].shape)
        post_view_count = slice_out_shape[dim]
        if post_view_count == 0:
            return False

        # Row-major stride of the sliced viewed dim, and of every pre-view dim.
        post_view_stride = prod(post_view_shape[dim + 1 :])
        pre_view_strides = self._row_major_strides(pre_view_shape)

        derived = self._derive_pre_view_slice(
            pre_view_shape,
            pre_view_strides,
            post_view_stride,
            post_view_size,
            start,
            stop,
            step,
            post_view_count,
        )
        if derived is None:
            return False
        pre_view_dim, pre_view_start, pre_view_stop, pre_view_step = derived

        graph = node.graph
        with graph.inserting_before(node):
            new_slice_args = (
                x_node,
                pre_view_dim,
                pre_view_start,
                pre_view_stop,
                pre_view_step,
            )
            new_slice = graph.create_node(
                "call_function",
                exir_ops.edge.aten.slice_copy.Tensor,
                args=new_slice_args,
            )
            new_slice.meta["val"] = exir_ops.edge.aten.slice_copy.Tensor(
                x_node.meta["val"], *new_slice_args[1:]
            )
            new_view = graph.create_node(
                "call_function",
                exir_ops.edge.aten.view_copy.default,
                args=(new_slice, list(slice_out_shape)),
            )
            new_view.meta["val"] = exir_ops.edge.aten.view_copy.default(
                new_slice.meta["val"], list(slice_out_shape)
            )

        node.replace_all_uses_with(new_view)
        return True

    @staticmethod
    def _row_major_strides(shape: tuple[int, ...]) -> list[int]:
        """Row-major (contiguous) strides for ``shape``."""
        strides = [1] * len(shape)
        acc = 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = acc
            acc *= shape[i]
        return strides

    def _normalize_slice(
        self, node: torch.fx.Node, post_view_size: int
    ) -> Optional[tuple[int, int, int]]:
        """Resolve the slice to concrete, clamped ``(start, stop, step)`` ints, or
        None if the bounds are dynamic or the step is non-positive (neither of
        which this pass handles)."""
        step = get_arg(node, "step")

        if not isinstance(step, int):
            return None

        if step <= 0:
            return None

        raw_start = get_arg(node, "start")
        raw_stop = get_arg(node, "end")

        # Make sure raw_start/raw_stop are not symbolic.
        if (raw_start is not None and not isinstance(raw_start, int)) or (
            raw_stop is not None and not isinstance(raw_stop, int)
        ):
            return None

        start = 0 if raw_start is None else raw_start
        stop = post_view_size if raw_stop is None else raw_stop
        if start < 0:
            start += post_view_size
        if stop < 0:
            stop += post_view_size
        start = max(0, min(start, post_view_size))
        stop = max(0, min(stop, post_view_size))
        return start, stop, step

    def _derive_pre_view_slice(
        self,
        pre_view_shape: tuple[int, ...],
        pre_view_strides: list[int],
        post_view_stride: int,
        post_view_size: int,
        start: int,
        stop: int,
        step: int,
        post_view_count: int,
    ) -> tuple[int, int, int, int] | None:
        """Return ``(dim, start, stop, step)`` for the single pre-view-tensor slice
        equivalent to slicing the viewed dim, or None if no single pre-view slice
        reproduces it.

        Both shapes index the same row-major flat space, so the sliced viewed dim
        (size ``post_view_size``, inner stride ``post_view_stride``) lines up with
        one pre-view dim (size ``pre_view_size``, inner stride ``pre_view_stride``)
        in one of three ways.
        """
        for pre_view_dim, (pre_view_stride, pre_view_size) in enumerate(
            zip(pre_view_strides, pre_view_shape)
        ):
            # Untouched: the viewed dim is identical to this pre-view dim (same
            # size and same inner stride), so the slice applies verbatim, any step.
            if pre_view_stride == post_view_stride and pre_view_size == post_view_size:
                return pre_view_dim, start, stop, step

            # Contiguous: the viewed dim and this pre-view dim span the same flat
            # extent (same period), and the selected band aligns to this dim's
            # boundaries. A contiguous (step==1) viewed slice
            # [start, start+post_view_count) is the flat band [start*
            # post_view_stride, (start+post_view_count)*post_view_stride), a
            # contiguous slice on this pre-view dim iff both ends are multiples of
            # its stride.
            if (
                step == 1
                and post_view_size * post_view_stride == pre_view_size * pre_view_stride
            ):
                flat_start = start * post_view_stride
                flat_stop = (start + post_view_count) * post_view_stride
                if (
                    flat_start % pre_view_stride == 0
                    and flat_stop % pre_view_stride == 0
                ):
                    return (
                        pre_view_dim,
                        flat_start // pre_view_stride,
                        flat_stop // pre_view_stride,
                        1,
                    )

            # Strided is the ONLY way the reshape itself introduces a stride, and
            # it requires a width-1 selection (post_view_count == 1): the viewed
            # dim is an innermost factor of this pre-view dim (identical inner
            # stride), so fixing that single factor index and letting the rest of
            # the pre-view dim run yields a uniform stride equal to the viewed dim's
            # size. Any wider selection (post_view_count > 1) of an inner factor
            # leaves runs separated by gaps -- block-strided, not a single slice --
            # so width-1 is required.
            if (
                post_view_count == 1
                and post_view_size > 1
                and pre_view_stride == post_view_stride
                and pre_view_size % post_view_size == 0
            ):
                pre_view_count = pre_view_size // post_view_size
                pre_view_stop = start + (pre_view_count - 1) * post_view_size + 1
                return pre_view_dim, start, pre_view_stop, post_view_size
        return None
