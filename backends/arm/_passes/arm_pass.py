# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import traceback
from abc import abstractmethod
from collections.abc import Collection
from typing import Any, List, Optional, Set, Type

import torch
from executorch.backends.arm._passes.mutable_buffer_utils import (
    collect_mutable_buffer_infos,
    restore_mutable_buffer_targets,
)
from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils import _pytree as pytree


_OPS_WITHOUT_QUANTIZED_FAKE_KERNEL = (
    exir_ops.edge.aten.bmm.default,
    exir_ops.edge.aten.leaky_relu.default,
)


class ArmPass(ExportPass):
    """Base class for Arm passes."""

    def __init__(self, tfa_pass: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.submodule_depth = 0
        self.is_tfa_pass = tfa_pass

    def allowed_to_transform(self, meta: NodeMetadata | dict[str, Any]) -> bool:
        if not self.is_tfa_pass:
            return True

        if isinstance(meta, NodeMetadata):
            meta_dict = meta.data
        else:
            meta_dict = meta

        disallow_tfa = meta_dict.get(DISALLOW_TFA_META_KEY, False)

        return not disallow_tfa

    def _is_quantized_meta(self, meta: NodeMetadata | dict[str, Any]) -> bool:
        """Return True when meta indicates fully quantized inputs and
        outputs.
        """
        if isinstance(meta, NodeMetadata):
            meta_dict = meta.data
        else:
            meta_dict = meta
        input_qparams = meta_dict.get("input_qparams", {})
        output_qparams = meta_dict.get("output_qparams", {})
        return bool(input_qparams) and bool(output_qparams)

    def should_fast_copy_node(self, target: torch.fx.node.Target) -> bool:
        # Metadata is unavailable here, so conservatively replay every op whose
        # quantized path needs a custom fake result.
        if any(target is op for op in _OPS_WITHOUT_QUANTIZED_FAKE_KERNEL):
            return False
        return super().should_fast_copy_node(target)

    @property
    @abstractmethod
    def _passes_required_after(self) -> Set[Type[ExportPass]]:
        """The subclass defines passes that must run after it."""
        pass

    @staticmethod
    def get_required_passes(pass_) -> List[str]:
        """Returns the list of passes that must be run after this pass, sorted
        by name.
        """
        if hasattr(pass_, "_passes_required_after"):
            return sorted([ArmPass.get_name(p) for p in pass_._passes_required_after])
        else:
            return []

    @staticmethod
    def get_name(pass_) -> str:
        """Returns the name of the pass."""
        if isinstance(pass_, ExportPass):
            return pass_.__class__.__name__
        elif hasattr(pass_, "__name__"):
            return pass_.__name__
        else:
            raise ValueError(
                f"Cannot get name for pass: {pass_}. It must be an instance of ExportPass or have a __name__ attribute."
            )

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
        if (
            op in _OPS_WITHOUT_QUANTIZED_FAKE_KERNEL
            and isinstance(meta, NodeMetadata)
            and len(meta.data.get("input_qparams", {})) > 0
        ):
            return self._call_quantized_op_without_fake_kernel(op, args, kwargs, meta)

        if not updated:
            return super().call_operator(op, args, kwargs, meta)

        # if updated we should update metadata
        new_meta = {}
        keys = meta.data.keys()
        for key in keys:
            new_meta[key] = meta[key]
        old_stack_trace = new_meta.get("stack_trace", "")
        new_meta["stack_trace"] = f"{old_stack_trace}\n{traceback.format_stack()[-2]}"
        return super().call_operator(op, args, kwargs, NodeMetadata(new_meta))

    def _call_quantized_op_without_fake_kernel(
        self,
        op,
        args: tuple[ProxyValue, ...],
        kwargs: dict[str, Any],
        meta: NodeMetadata,
    ) -> ProxyValue:
        old_val = meta.data["val"]
        output_qparams = meta.data.get("output_qparams", {})
        dtype = (
            next(iter(output_qparams.values())).dtype
            if len(output_qparams) > 0
            else old_val.dtype
        )
        res_data = torch.empty_like(old_val, dtype=dtype)

        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )
        res_proxy = self.tracer.create_proxy(
            "call_function",
            op,
            args_proxy,
            kwargs_proxy,
        )
        res_proxy.node.meta.update(meta.data)
        self.tracer.set_metadata(res_proxy.node, res_data)
        return ProxyValue(res_data, res_proxy)

    def should_run_on_nested_submodule(self, graph_module: GraphModule) -> bool:
        """Return whether subclass rewrites should run in this nested submodule."""
        has_custom_precheck = type(self).should_run_pass is not ArmPass.should_run_pass
        return has_custom_precheck and self.should_run_pass(graph_module)

    def call_submodule(
        self, graph_module: GraphModule, inputs: tuple[Any, ...]
    ) -> PassResult:
        self.submodule_depth += 1
        try:
            if self.submodule_depth == 1:
                self._top_level_call_operator = self.call_operator
                return super().call_submodule(graph_module, inputs)

            call_operator = self.call_operator
            try:
                if self.should_run_on_nested_submodule(graph_module):
                    self.call_operator = self._top_level_call_operator  # type: ignore
                else:
                    # Nested submodules still need replay without subclass rewrites.
                    self.call_operator = super().call_operator  # type: ignore
                return super().call_submodule(graph_module, inputs)
            finally:
                self.call_operator = call_operator  # type: ignore
        finally:
            self.submodule_depth -= 1

    def call_shape_operator(
        self, op, args: tuple, kwargs: dict, meta: NodeMetadata, updated: bool = True
    ) -> ProxyValue:
        """Call operator for shape-producing operators.

        This function is responsible for marking the output of the operator with
        the TosaSpecialDtype of SHAPE, so that later passes can identify it as a
        shape-producing operator and handle it accordingly.

        """
        # Copy meta and set TosaSpecialDtype to SHAPE
        if not isinstance(meta, NodeMetadata):
            raise TypeError("Expected meta to be of type NodeMetadata")
        shape_meta = copy.copy(meta)
        shape_meta.data = dict(meta.data)
        shape_meta.data[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE
        # Call the super (ArmPass) call operator with updated meta
        return self.call_operator(op, args, kwargs, shape_meta, updated)

    def call_scalar(self, value: int | float, meta: NodeMetadata | dict[str, Any]):
        """Return a scalar value for the current pass stage.

        In transform-for-annotation passes this returns the Python scalar
        directly. In later passes it materializes a `(1,)` `aten.full` node
        using the output dtype/device from `meta["val"]` when available.

        """

        if self.is_tfa_pass:
            return value

        kwargs = {}
        if "val" in meta:
            val = meta["val"]
            if isinstance(val, tuple):
                val = val[0]
            kwargs = {"device": val.device, "dtype": val.dtype}

        return ArmPass.call_operator(
            self,
            op=exir_ops.edge.aten.full.default,
            args=((1,), value),
            kwargs=kwargs,
            meta=meta,
            updated=True,
        )

    def should_run_pass(self, graph_module: GraphModule) -> bool:
        """Return whether this pass should run on the graph module.

        Subclasses can override this to cheaply skip the pass before
        ``call()`` starts the normal ``ExportPass`` retracing path.

        Args:
            graph_module (GraphModule): The graph module to inspect.

        Returns:
            bool: True when the pass should run.

        """
        return True

    def __call__(self, graph_module: GraphModule) -> PassResult | None:
        self.requires(graph_module)
        if not self.should_run_pass(graph_module):
            self.ensures(graph_module)
            return PassResult(graph_module, False)
        mutable_buffers = collect_mutable_buffer_infos(graph_module)
        res = self.call(graph_module)
        self.ensures(graph_module)
        if res is not None:
            res = PassResult(
                restore_mutable_buffer_targets(res.graph_module, mutable_buffers),
                res.modified,
            )
        return res


class ArmOpTargetedPass(ArmPass):
    """Base class for passes that only transform selected operators.

    Subclasses must set ``target_ops`` to the exhaustive set of call_function
    targets their ``call_operator()`` can transform. This ARM-specific contract
    drives both the target pre-scan and fast-copy eligibility; an empty
    ``target_ops`` disables fast copy and skips the pass. Generic ``ExportPass``
    subclasses instead use ``targeted_ops`` for explicit fast-copy opt-in.

    If the current graph and nested control-flow subgraphs do not contain any
    target, the pass returns immediately without paying the default ExportPass
    retracing cost.

    Set ``check_allowed_to_transform`` to ``True`` when the target pre-scan
    should also apply ``allowed_to_transform()`` to matching target nodes. This
    is useful for TFA passes whose ``call_operator()`` leaves disallowed target
    nodes unchanged. If all matching targets are disallowed, the pass can
    return before entering the normal ``ExportPass`` path.

    """

    enable_fast_copy = True
    target_ops: Collection[Any] = ()
    check_allowed_to_transform = False

    def get_fast_copy_target_ops(self) -> Optional[tuple[Any, ...]]:
        if not self.enable_fast_copy:
            return None
        targets = tuple(self.target_ops)
        return targets if targets else None

    def should_run_on_nested_submodule(self, graph_module: GraphModule) -> bool:
        return self.has_target_node(graph_module, recursive=False)

    def has_target_node(
        self, graph_module: GraphModule, recursive: bool = True
    ) -> bool:
        """Return whether the graph module tree contains a target node.

        Args:
            graph_module (GraphModule): The graph module tree to inspect.
            recursive (bool): Whether to inspect nested child GraphModules.
                When false, inspect only ``graph_module`` itself.

        Returns:
            bool: True if a matching call_function node is present.

        """
        visited_graph_modules = set()

        def target_node_can_trigger_pass(node: Node) -> bool:
            if not self.check_allowed_to_transform:
                return True
            if self.allowed_to_transform(node.meta):
                return True
            return False

        def graph_has_target(module: GraphModule) -> bool:
            if id(module) in visited_graph_modules:
                return False
            visited_graph_modules.add(id(module))

            for target in self.target_ops:
                for node in module.graph.find_nodes(
                    op="call_function",
                    target=target,
                    sort=False,
                ):
                    if target_node_can_trigger_pass(node):
                        return True

            return recursive and any(
                isinstance(child, GraphModule) and graph_has_target(child)
                for child in module.children()
            )

        return graph_has_target(graph_module)

    def should_run_pass(self, graph_module: GraphModule) -> bool:
        """Return whether this pass has a target node to transform.

        Args:
            graph_module (GraphModule): The graph module tree to inspect.

        Returns:
            bool: True when a matching target node is present.

        """
        return self.has_target_node(graph_module)
