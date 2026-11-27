# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.transforms import remove_getitem_op
from executorch.exir.pass_base import ExportPass


class RemoveGetItemPass(ArmPass, remove_getitem_op.RemoveGetItemPass):
    _passes_required_after: Set[Type[ExportPass]] = set()
    _target_names = {
        "aten.max_pool2d_with_indices.default",
        "aten.max.dim",
    }

    def should_run_pass(self, graph_module) -> bool:
        return any(
            node.op == "call_function"
            and getattr(node.target, "__name__", None) in self._target_names
            for node in graph_module.graph.nodes
        )
