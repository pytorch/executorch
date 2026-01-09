# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import traceback
from abc import abstractmethod
from typing import Any, List, Optional, Set, Type

from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.exir.pass_base import ExportPass, NodeMetadata
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class ArmPass(ExportPass):
    """Base class for Arm passes"""

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

    @property
    @abstractmethod
    def _passes_required_after(self) -> Set[Type[ExportPass]]:
        """The subclass defines passes that must run after it"""
        pass

    @staticmethod
    def get_required_passes(pass_) -> List[str]:
        """
        Returns the list of passes that must be run after this pass, sorted by name.
        """
        if hasattr(pass_, "_passes_required_after"):
            return sorted([ArmPass.get_name(p) for p in pass_._passes_required_after])
        else:
            return []

    @staticmethod
    def get_name(pass_) -> str:
        """
        Returns the name of the pass.
        """
        if isinstance(pass_, ExportPass):
            return pass_.__class__.__name__
        elif hasattr(pass_, "__name__"):
            return pass_.__name__
        else:
            raise ValueError(
                f"Cannot get name for pass: {pass_}. It must be an instance of ExportPass or have a __name__ attribute."
            )

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
