# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
from typing import Any, List, Tuple

import torch
from executorch.exir import memory

from executorch.exir.dialects._ops import ops
from executorch.exir.tensor import (
    contiguous_stride_from_shape,
    determine_tensor_dynanism,
    dim_order_from_stride,
    TensorShapeDynamism,
    TensorSpec,
)
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)


def _is_view_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.view_copy.default,
        ops.edge.aten.view_copy.default,
    )


def _is_select_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.select_copy.int,
        ops.edge.aten.select_copy.int,
    )


_VIEW_OP = memory.view
_SELECT_OP = memory.select



class _Guard:
    def __init__(
        self, name: str, field_lambda, expected_val: Any  # pyre-ignore[2]
    ) -> None:
        self.name: str = name
        self.field_lambda = field_lambda  # pyre-ignore[4]
        self.expected_val = copy.deepcopy(expected_val)  # pyre-ignore[4]

    def __call__(self, view_spec) -> None:  # pyre-ignore[2]
        assert view_spec._unguarded_access
        observed_val = self.field_lambda(view_spec)
        if observed_val != self.expected_val:
            raise Exception(
                f"Guard {self.name} failed.  Expected to see value {self.expected_val}, but saw value {observed_val}."
            )


class _ViewSpec(TensorSpec):
    def __init__(
        self,
        base: TensorSpec,
        shape: List[int],
        byte_offset: int = 0,
    ) -> None:
        """
        A _ViewSpec is TensorSpec that shares non-size related fields with its base.
        The size-related fields are: shape, stride, dim_order, and shape_dynamism.

        If either the base or view spec updates a non-size related field, the change
        is reflected in both specs.  But size related fields are not linked and can
        be set separately.

        A _ViewSpec can only be created from a non-sparse, strided TensorSpec.
        On creation, a _ViewSpec must be compatible with its base with respect to
        shape_dynamism, dtype, and nbytes (when byte_offset is 0).

        When byte_offset is non-zero (used for select/slice sub-views), the view
        describes a contiguous sub-region of the base at the given byte offset.
        In this case, nbytes may differ from the base and rank may change.

        A _ViewSpec contains _guards that are evaluated on every __getattribute__ call.
        The purpose of the guards is to make sure the _ViewSpec is still compatible
        with its base.
        """

        # Explicitly put all attributes into _self_fields or _base_fields
        # Any attribute that is not in _self_fields or _base_fields will
        # raise an Exception.  If TensorSpec is extended with a new attribute,
        # we should explicitly decide how _ViewSpec will handle it.
        self._self_fields = [
            # We need to get the debug method from self
            # so that the object id it prints is correct.
            "debug",  # method
            "__repr__",  # method
            # The following are related to size and should use self
            "shape",
            "stride",
            "dim_order",
            "shape_dynamism",
            "nbytes",  # method
            "allocated_memory",  # property
            "is_dynamic_shape_tensor",  # property
            "is_static_shape_tensor",  # property
            "is_upper_bound_tensor",  # property
            "is_dynamic_unbound_tensor",  # property
        ]
        self._base_fields = [
            "scalar_type",
            "const",
            "alignment",
            "storage",
            "requires_grad",
            "layout",
            "is_sparse",
            "init_mem_planning_fields",  # method
            "realign",  # method
            "from_tensor",  # class method
            "lifetime",
            "mem_id",
            "mem_obj_id",
            "mem_offset",
            "dtype",  # property
            "extra_tensor_info",  # property
            "device",
            "device_index",
        ]

        # Make sure _self_fields and _base_fields are disjoint
        assert len(set(self._self_fields) & set(self._base_fields)) == 0

        self._guards: List[_Guard] = []
        self._unguarded_access = False
        self._byte_offset: int = byte_offset

        # Make sure base is not sparse and add a guard
        if base.is_sparse:
            raise Exception(
                "_ViewSpec can only be created from non-sparse TensorSpec, but base.is_sparse=True."
            )
        self._guards.append(
            _Guard(
                "is_sparse",
                lambda view_spec: view_spec.is_sparse,
                False,
            )
        )

        # Make sure base layout is strided and add a guard
        if base.layout != torch.strided:
            raise Exception(
                f"_ViewSpec can only be created from TensorSpec with layout={torch.strided}, but got layout={base.layout}."
            )
        self._guards.append(
            _Guard(
                "layout",
                lambda view_spec: view_spec.layout,
                torch.strided,
            )
        )

        self._base = base
        self.shape: List[int] = shape
        self.stride: Tuple[int] = contiguous_stride_from_shape(torch.Size(self.shape))
        self.dim_order: Tuple[bytes] = dim_order_from_stride(self.stride)
        self.shape_dynamism: TensorShapeDynamism = determine_tensor_dynanism(
            torch.Size(self.shape)
        )

        if self.dtype != base.dtype:
            raise Exception(
                f"_ViewSpec is incompatible with its base on creation.  It has dtype={self.dtype}, but its base has dtype={base.dtype}."
            )
        self._guards.append(
            _Guard("dtype", lambda view_spec: view_spec.dtype, base.dtype)
        )

        # For full views (same nbytes, zero offset), dynamism and rank must match.
        # For sub-views (select/slice), the output is a contiguous subset so
        # nbytes will differ, rank may change, and dynamism may differ
        # (e.g., selecting away a dynamic dim produces a static output).
        is_full_view = byte_offset == 0 and self.nbytes() == base.nbytes()
        if is_full_view:
            if self.shape_dynamism != base.shape_dynamism:
                raise Exception(
                    f"_ViewSpec is incompatible with its base on creation.  It has shape_dynamism={self.shape_dynamism}, but its base has shape_dynamism={base.shape_dynamism}."
                )
            self._guards.append(
                _Guard(
                    "shape_dynamism_eq_base",
                    lambda view_spec: view_spec.shape_dynamism
                    == view_spec._base.shape_dynamism,
                    True,
                )
            )
            self._guards.append(
                _Guard("rank", lambda view_spec: len(view_spec.shape), len(shape))
            )
        else:
            if self.nbytes() + byte_offset > base.nbytes():
                raise Exception(
                    f"_ViewSpec sub-view extends beyond base.  Sub-view needs {self.nbytes()} bytes at offset {byte_offset}, but base has {base.nbytes()} bytes."
                )

    def _run_guards(self) -> None:
        unguarded_access = self._unguarded_access
        try:
            self._unguarded_access = True
            for g in self._guards:
                g(self)
        finally:
            self._unguarded_access = unguarded_access

    def __getattribute__(self, name: str):  # pyre-ignore
        # Special field so we don't recurse infinitely
        if name in [
            "_base",
            "_self_fields",
            "_base_fields",
            "_guards",
            "_unguarded_access",
            "_run_guards",
            "_byte_offset",
        ]:
            return object.__getattribute__(self, name)

        # Get some attributes from self
        if name in self._self_fields:
            val = object.__getattribute__(self, name)
        elif name in self._base_fields:
            val = getattr(self._base, name)
            # For static sub-views, adjust mem_offset by byte_offset so the
            # emitter can elide the op. For non-static sub-views, return
            # None for mem_id/mem_offset — et_select runs at runtime and
            # the output needs no allocation info.
            if name in ("mem_id", "mem_offset") and val is not None:
                byte_offset = object.__getattribute__(self, "_byte_offset")
                if byte_offset != 0:
                    shape_dynamism = object.__getattribute__(self, "shape_dynamism")
                    if shape_dynamism != TensorShapeDynamism.STATIC:
                        val = None
                    elif name == "mem_offset":
                        val = val + byte_offset
        else:
            if len(name) > 0 and name[0] != "_":
                logger.warning(
                    f"Getting non-private attribute {name} on self, but it is not in _self_fields or _base_fields.  Is this intended?"
                )
            val = object.__getattribute__(self, name)

        if not self._unguarded_access:
            self._run_guards()
        return val

    def __setattr__(self, name: str, val) -> None:  # pyre-ignore
        # Special field so we don't recurse infinitely
        if name in [
            "_base",
            "_self_fields",
            "_base_fields",
            "_guards",
            "_unguarded_access",
            "_run_guards",
            "_byte_offset",
        ]:
            object.__setattr__(self, name, val)
            return

        if name in self._self_fields:
            object.__setattr__(self, name, val)
            return

        if name in self._base_fields:
            object.__setattr__(self._base, name, val)
            return

        if len(name) > 0 and name[0] != "_":
            logger.warning(
                f"Setting non-private attribute {name} on self, but it is not in _self_fields or _base_fields.  Is this intended?"
            )
        object.__setattr__(self, name, val)


class ReplaceViewCopyWithViewPass(PassBase):
    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """
        This pass replaces view_copy nodes with view nodes.

        This should be run after the NormalizeViewCopyBasePass.

        During memory planning, view nodes share the same storage as their base.
        """

        n_replaced = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                # Note: We only replace view_copy nodes that are not output, since
                # the output pointer could be modified at runtime (T187925929)
                if _is_view_copy(node) and all(u.op != "output" for u in node.users):
                    base, _ = node.args
                    node.target = _VIEW_OP

                    # Create spec for the node.
                    # _ViewSpec gives a view into its base spec for non-size
                    # related information.

                    # the shape is not the same as node.args[1] because node.args[1]
                    # can have an inferred sizes (-1).
                    shape = node.meta["val"].shape
                    node.meta["spec"] = _ViewSpec(base.meta["spec"], shape)

                    n_replaced += 1

                elif _is_select_copy(node) and all(
                    u.op != "output" for u in node.users
                ):
                    replaced = self._try_replace_select(node)
                    if replaced:
                        n_replaced += 1

            module.recompile()

        logger.debug(f"Replaced {n_replaced} view_copy nodes with {_VIEW_OP} nodes.")
        logger.debug(
            f"Replaced {n_replaced} select_copy nodes with {_SELECT_OP} nodes."
        )
        return PassResult(graph_module, n_replaced > 0)

    def _try_replace_select(self, node: torch.fx.Node) -> bool:
        base = node.args[0]
        assert isinstance(base, torch.fx.Node)
        base_spec = base.meta["spec"]

        dim: int = node.args[1]
        index: int = node.args[2]
        base_shape = list(base_spec.shape)

        if dim < 0:
            dim += len(base_shape)

        if any(base_shape[i] != 1 for i in range(dim)):
            return False

        if index < 0:
            index += base_shape[dim]

        base_stride = contiguous_stride_from_shape(torch.Size(base_shape))
        element_size = torch._utils._element_size(base_spec.dtype)
        byte_offset = index * base_stride[dim] * element_size

        if base_spec.const:
            return False

        node.target = _SELECT_OP
        view_spec = _ViewSpec(base_spec, node.meta["val"].shape, byte_offset)
        if base_spec.shape_dynamism != TensorShapeDynamism.STATIC:
            view_spec.shape_dynamism = base_spec.shape_dynamism
        node.meta["spec"] = view_spec
        return True

    def ensures(self, graph_module: torch.fx.GraphModule) -> None:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                # Note: We only replace view_copy nodes that are not output, since
                # the output pointer could be modified at runtime (T187925929)
                assert not (
                    _is_view_copy(node) and all(u.op != "output" for u in node.users)
                )
                if node.op == "call_function" and node.target == _VIEW_OP:
                    assert isinstance(node.meta["spec"], _ViewSpec)
                if node.op == "call_function" and node.target == _SELECT_OP:
                    assert isinstance(node.meta["spec"], _ViewSpec)

    def requires(self, graph_module: torch.fx.GraphModule) -> None:
        """
        This pass should be called after NormalizeViewCopyBasePass.
        We check that all view_copy nodes have been normalized.
        """
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                # Note: We only replace view_copy nodes that are not output, since
                # the output pointer could be modified at runtime (T187925929)
                if _is_view_copy(node) and all(u.op != "output" for u in node.users):
                    base, size = node.args
                    assert not _is_view_copy(base)
