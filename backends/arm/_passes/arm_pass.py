# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import traceback
from abc import abstractmethod
from typing import List, Optional, Set, Type

from executorch.exir.pass_base import ExportPass, NodeMetadata


class ArmPass(ExportPass):
    """Base class for Arm passes"""

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
