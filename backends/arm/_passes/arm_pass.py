# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import traceback
from abc import abstractmethod
from typing import Any, List, Optional, Set, Type

from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class ArmPass(ExportPass):
    """Base class for Arm passes."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "targeted_ops", None) is not None:
            return
        # Only auto-discover targeted_ops for passes that use the standard
        # call_operator() pattern. Passes that override call() use _TARGET_OPS
        # for their own graph manipulation logic, not as a fast-copy declaration.
        if "call" in cls.__dict__:
            return
        for attr in ("_TARGET_OPS", "_supported_ops"):
            ops = getattr(cls, attr, None)
            if ops:
                cls.targeted_ops = set(ops) if not isinstance(ops, set) else ops  # type: ignore[attr-defined]
                return
        edge = getattr(cls, "_EDGE_OPS", None)
        aten = getattr(cls, "_ATEN_OPS", None)
        if edge or aten:
            cls.targeted_ops = {*(edge or ()), *(aten or ())}  # type: ignore[attr-defined]

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

    def should_run(self, graph_module: GraphModule) -> bool:
        """Skip this pass if the graph contains none of its targeted ops.

        Subclasses that define a ``targeted_ops`` class attribute (a set of
        op overloads) get this check for free via inheritance.  Passes
        without ``targeted_ops`` always run (the default).

        Recursively checks control flow submodules (cond/while_loop) so
        passes are not incorrectly skipped when targeted ops are nested.

        """
        targeted = getattr(self, "targeted_ops", None)
        if targeted is None:
            return True

        from executorch.exir.graph_module import get_control_flow_submodules

        def _has_targeted_op(gm: GraphModule) -> bool:
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target in targeted:
                    return True
            for _, submod, _ in get_control_flow_submodules(gm):
                if _has_targeted_op(submod):
                    return True
            return False

        return _has_targeted_op(graph_module)

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
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

    def call_submodule(
        self, graph_module: GraphModule, inputs: tuple[Any, ...]
    ) -> PassResult:
        self.submodule_depth += 1
        if self.submodule_depth == 1:
            result = super().call_submodule(graph_module, inputs)
        else:
            # When we trace a submodule, we don't want to apply the calling pass.
            # Temporarily replace call_operator to avoid this.
            _call_operator_fn = self.call_operator
            self.call_operator = super().call_operator  # type: ignore
            result = super().call_submodule(graph_module, inputs)
            self.call_operator = _call_operator_fn  # type: ignore
        self.submodule_depth -= 1
        return result

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
