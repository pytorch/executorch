# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Replace safe, static, contiguous slice copies with sub-buffer aliases."""

import logging
from typing import Any, Optional

import torch
from executorch.exir import memory
from executorch.exir.dialects._ops import ops
from executorch.exir.passes.replace_view_copy_with_view_pass import (
    _ViewSpec,
    is_copy_to_view_safe,
)
from executorch.exir.tensor import (
    contiguous_stride_from_shape,
    dim_order_from_stride,
    TensorSpec,
)
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)


def _is_slice_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.slice_copy.Tensor,
        ops.edge.aten.slice_copy.Tensor,
    )


def _is_static_slice_argument(value: Any) -> bool:
    return value is None or isinstance(value, int)


def is_contiguous_slice_copy(node: torch.fx.Node) -> bool:
    """True for static-argument, outermost-dimension, unit-step slices."""
    if (
        not _is_slice_copy(node)
        or node.kwargs
        or not all(_is_static_slice_argument(arg) for arg in node.args[1:])
    ):
        return False
    dim = node.args[1] if len(node.args) > 1 else 0
    step = node.args[4] if len(node.args) > 4 else 1
    base = node.args[0]
    val = base.meta.get("val") if isinstance(base, torch.fx.Node) else None
    if not isinstance(dim, int) or step != 1:
        return False
    if dim < 0 and isinstance(val, torch.Tensor):
        dim += val.dim()
    return dim == 0


def _compute_slice_byte_offset(base: TensorSpec, dim: int, start: Optional[int]) -> int:
    start, _, _ = slice(start, None, 1).indices(base.shape[dim])
    return start * base.stride[dim] * torch._utils._element_size(base.dtype)


def _static_slice_bounds(node: torch.fx.Node) -> Optional[tuple[int, int]]:
    base = node.args[0]
    if not isinstance(base, torch.fx.Node) or base.op == "placeholder":
        return None
    # Retain copies of aliases, including pending slice candidates.
    if _is_slice_copy(base) or base.target in (memory.slice, memory.view):
        return None
    base_spec = base.meta.get("spec")
    output_spec = node.meta.get("spec")
    if (
        not isinstance(base_spec, TensorSpec)
        or not isinstance(output_spec, TensorSpec)
        or not base_spec.is_static_shape_tensor
        or not output_spec.is_static_shape_tensor
        or output_spec.dtype != base_spec.dtype
        or base_spec.const
        or base_spec.is_sparse
        or base_spec.layout != torch.strided
        or base_spec.dim_order
        != dim_order_from_stride(
            contiguous_stride_from_shape(torch.Size(base_spec.shape))
        )
    ):
        return None
    start = node.args[2] if len(node.args) > 2 else None
    end = node.args[3] if len(node.args) > 3 else None
    start, end, _ = slice(start, end, 1).indices(base_spec.shape[0])
    shape = [max(end - start, 0), *base_spec.shape[1:]]
    # Empty results retain their kernel; do not emit an alias at
    # the end of (or into an empty) allocation.
    if 0 in shape or list(output_spec.shape) != shape:
        return None
    return start, end


class ReplaceSliceCopyWithSlicePass(PassBase):
    """Replace eligible slice copies after view-copy replacement."""

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        n_replaced = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            replacements = {}
            # Analyze consumers first, with already-decided aliasing behavior.
            # Specs are rebuilt in forward order below so views of slices use
            # the final base spec, including its byte offset.
            for node in reversed(module.graph.nodes):
                if not is_contiguous_slice_copy(node) or any(
                    u.op == "output" for u in node.users
                ):
                    continue
                bounds = _static_slice_bounds(node)
                if bounds is None:
                    continue
                start, end = bounds
                base = node.args[0]
                base_spec = base.meta["spec"]
                if not is_copy_to_view_safe(node, (memory.view, memory.slice)):
                    continue
                replacements[node] = _compute_slice_byte_offset(base_spec, 0, start)
                node.target = memory.slice
                node.args = (base, 0, start, end, 1)
                n_replaced += 1

            updated_specs = set()
            for node in module.graph.nodes:
                if node in replacements:
                    node.meta["spec"] = _ViewSpec(
                        node.args[0].meta["spec"],
                        list(node.meta["spec"].shape),
                        byte_offset=replacements[node],
                    )
                    updated_specs.add(node)
                elif node.target == memory.view and node.args[0] in updated_specs:
                    node.meta["spec"] = _ViewSpec(
                        node.args[0].meta["spec"], list(node.meta["spec"].shape)
                    )
                    updated_specs.add(node)
            module.recompile()

        logger.debug("Replaced %d slice_copy nodes with memory.slice", n_replaced)
        return PassResult(graph_module, n_replaced > 0)

    def ensures(self, graph_module: torch.fx.GraphModule) -> None:
        for module in graph_module.modules():
            if isinstance(module, torch.fx.GraphModule):
                for node in module.graph.nodes:
                    if node.target == memory.slice:
                        assert isinstance(node.meta["spec"], _ViewSpec)
