# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Re-inplace contiguous ``slice_copy`` nodes as lightweight slices (#10917).

Slice analog of :class:`ReplaceViewCopyWithViewPass`.  Contiguous slices taken
along the outermost dimension with unit step alias a sub-region of the base
buffer and can be represented with :class:`_SliceSpec`, which shares the
base's ``mem_id`` but uses a computed byte ``mem_offset``.
"""

import logging
from typing import Any, List, Optional

import torch
from executorch.exir import memory
from executorch.exir.dialects._ops import ops
from executorch.exir.sym_util import eval_shape
from executorch.exir.tensor import (
    contiguous_stride_from_shape,
    determine_tensor_dynanism,
    dim_order_from_stride,
    TensorSpec,
)
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)

_SLICE_OP = memory.slice


def _is_slice_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.slice_copy.Tensor,
        ops.edge.aten.slice_copy.Tensor,
    )


def _normalize_dim(dim: int, rank: Optional[int]) -> Optional[int]:
    if isinstance(dim, int) and dim < 0:
        if rank is None:
            return None
        dim = dim + rank
    return dim


def is_contiguous_slice_copy(node: torch.fx.Node) -> bool:
    """True if ``node`` is an outermost-dim, unit-step ``slice_copy``."""
    if not _is_slice_copy(node):
        return False

    args = node.args
    self_arg = args[0]
    dim = args[1] if len(args) > 1 else 0
    step = args[4] if len(args) > 4 else 1

    if step != 1:
        return False

    rank = None
    self_val = self_arg.meta.get("val") if isinstance(self_arg, torch.fx.Node) else None
    if self_val is not None and hasattr(self_val, "dim"):
        rank = self_val.dim()

    dim = _normalize_dim(dim, rank)
    return dim == 0


def _slice_start_as_int(start: Any) -> int:
    if start is None:
        return 0
    if isinstance(start, int):
        return start
    if isinstance(start, torch.SymInt):
        return int(eval_shape([start])[0])
    return int(start)


def _compute_slice_byte_offset(base: TensorSpec, dim: int, start: Any) -> int:
    start_int = _slice_start_as_int(start)
    if start_int < 0:
        raise ValueError("memory.slice does not support negative slice starts.")
    elem_size = torch._utils._element_size(base.dtype)
    return start_int * base.stride[dim] * elem_size


def _is_aliasing_base(base: torch.fx.Node) -> bool:
    """Whether ``base`` is itself an alias rather than a real allocation.

    ``memory.slice`` and ``memory.view`` nodes do not own storage, so a slice
    taken from one has no concrete ``mem_offset`` to build on until the chain is
    normalized down to the first real allocation.
    """
    return base.op == "call_function" and base.target in (memory.slice, memory.view)


def _has_default_dim_order(spec: TensorSpec) -> bool:
    """Whether ``spec`` has the standard contiguous dimension ordering.

    ``_SliceSpec`` computes a contiguous output stride.  That is only a valid
    alias for a dim-0 slice when the base itself has the default dim order.
    """
    return spec.dim_order == dim_order_from_stride(
        contiguous_stride_from_shape(torch.Size(spec.shape))
    )


class _SliceSpec(TensorSpec):
    """TensorSpec for a zero-copy slice into a contiguous base buffer."""

    def __init__(
        self,
        base: TensorSpec,
        shape: List[int],
        dim: int,
        start: Any,
    ) -> None:
        if base.is_sparse:
            raise Exception(
                "_SliceSpec can only be created from non-sparse TensorSpec."
            )
        if base.layout != torch.strided:
            raise Exception(f"_SliceSpec requires strided layout, got {base.layout}.")

        self._base = base
        self._byte_offset = _compute_slice_byte_offset(base, dim, start)
        self._unguarded_access = False

        self._self_fields = [
            "debug",
            "__repr__",
            "shape",
            "stride",
            "dim_order",
            "shape_dynamism",
            "nbytes",
            "allocated_memory",
            "is_dynamic_shape_tensor",
            "is_static_shape_tensor",
            "is_upper_bound_tensor",
            "is_dynamic_unbound_tensor",
            "mem_offset",
        ]
        self._base_fields = [
            "scalar_type",
            "const",
            "alignment",
            "storage",
            "requires_grad",
            "layout",
            "is_sparse",
            "init_mem_planning_fields",
            "realign",
            "from_tensor",
            "lifetime",
            "mem_id",
            "mem_obj_id",
            "dtype",
            "extra_tensor_info",
            "device",
            "device_index",
            # Read by the memory planning algorithms (e.g. ``greedy``).  A slice
            # is never itself an in-place target, so it defers to its base.
            "inplace_base",
        ]

        self.shape = list(shape)
        self.stride = contiguous_stride_from_shape(torch.Size(self.shape))
        self.dim_order = dim_order_from_stride(self.stride)
        self.shape_dynamism = determine_tensor_dynanism(torch.Size(self.shape))

        if self.shape_dynamism != base.shape_dynamism:
            raise Exception(
                f"_SliceSpec shape_dynamism {self.shape_dynamism} != base {base.shape_dynamism}"
            )
        if self.dtype != base.dtype:
            raise Exception(f"_SliceSpec dtype {self.dtype} != base {base.dtype}")

    def __getattribute__(self, name: str):  # pyre-ignore
        if name in [
            "_base",
            "_self_fields",
            "_base_fields",
            "_byte_offset",
            "_unguarded_access",
        ]:
            return object.__getattribute__(self, name)

        self_fields = object.__getattribute__(self, "_self_fields")
        base_fields = object.__getattribute__(self, "_base_fields")

        if name == "mem_offset":
            base = object.__getattribute__(self, "_base")
            base_offset = base.mem_offset
            if base_offset is None:
                return None
            byte_offset = object.__getattribute__(self, "_byte_offset")
            return base_offset + byte_offset

        if name in self_fields:
            if name in ("nbytes", "allocated_memory"):
                return TensorSpec.__getattribute__(self, name)
            return object.__getattribute__(self, name)

        if name in base_fields:
            base = object.__getattribute__(self, "_base")
            return object.__getattribute__(base, name)

        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, val) -> None:  # pyre-ignore
        if name in [
            "_base",
            "_self_fields",
            "_base_fields",
            "_byte_offset",
            "_unguarded_access",
        ]:
            object.__setattr__(self, name, val)
            return

        if hasattr(self, "_self_fields") and name in self._self_fields:
            if name == "mem_offset":
                raise Exception("_SliceSpec.mem_offset is computed from the base.")
            object.__setattr__(self, name, val)
            return

        if hasattr(self, "_base_fields") and name in self._base_fields:
            object.__setattr__(self._base, name, val)
            return

        object.__setattr__(self, name, val)


class ReplaceSliceCopyWithSlicePass(PassBase):
    """Replace eligible contiguous ``slice_copy`` nodes with ``memory.slice``."""

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        n_replaced = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if is_contiguous_slice_copy(node) and all(
                    u.op != "output" for u in node.users
                ):
                    base = node.args[0]
                    if (
                        not isinstance(base, torch.fx.Node)
                        or "spec" not in base.meta
                        or not base.meta["spec"].is_static_shape_tensor
                        or not _has_default_dim_order(base.meta["spec"])
                    ):
                        # Specs are populated by the lowering pipeline before this
                        # pass.  Skip bare FX graphs so the pass remains safe to use
                        # in isolation as well.
                        continue
                    if _is_aliasing_base(base):
                        # The base is itself an alias (a slice or a view), so it
                        # has no allocation of its own to offset from.  Chaining
                        # offsets through it would require normalizing to the
                        # first real allocation first, so leave this as a copy.
                        continue
                    dim = node.args[1] if len(node.args) > 1 else 0
                    start = node.args[2] if len(node.args) > 2 else None
                    if _slice_start_as_int(start) < 0:
                        # Negative starts are relative to the end of the
                        # dimension.  They cannot be expressed as a static
                        # offset without normalizing against the base shape.
                        continue
                    node.target = _SLICE_OP
                    shape = node.meta["val"].shape
                    node.meta["spec"] = _SliceSpec(
                        base.meta["spec"], list(shape), dim, start
                    )
                    n_replaced += 1

            module.recompile()

        logger.debug(
            "ReplaceSliceCopyWithSlicePass: replaced %d slice_copy node(s) with %s.",
            n_replaced,
            _SLICE_OP,
        )
        return PassResult(graph_module, n_replaced > 0)

    def ensures(self, graph_module: torch.fx.GraphModule) -> None:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op == "call_function" and node.target == _SLICE_OP:
                    assert isinstance(node.meta["spec"], _SliceSpec)
