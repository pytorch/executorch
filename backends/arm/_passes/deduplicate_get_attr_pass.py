# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


class DeduplicateGetAttrPass(ArmPass):
    """Give duplicate get_attr nodes distinct backing attributes.

    Torchao's constant folder can delete a shared backing attribute while
    another get_attr node still refers to it. Keep separate graph nodes so PT2E
    can attach per-use observers and backend lowering can process constants per
    use.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def _get_attr(self, graph_module: GraphModule, target: str) -> Any:
        attr: Any = graph_module
        for target_atom in target.split("."):
            attr = getattr(attr, target_atom)
        return attr

    def _copy_attr(self, graph_module: GraphModule, node: Node) -> str:
        """Register a new attribute referring to the same data as the original
        one.
        """

        assert isinstance(node.target, str)
        attr = self._get_attr(graph_module, node.target)
        get_new_attr_name = get_new_attr_name_with_prefix(
            f"_deduplicated_get_attr_{node.name}_"
        )
        attr_name = get_new_attr_name(graph_module)

        if isinstance(attr, torch.nn.Parameter):
            graph_module.register_parameter(attr_name, attr)
        elif isinstance(attr, torch.Tensor):
            graph_module.register_buffer(attr_name, attr)
        else:
            setattr(graph_module, attr_name, attr)

        return attr_name

    def call(self, graph_module: GraphModule) -> PassResult:
        seen_targets: set[str] = set()
        modified = False

        for node in graph_module.graph.find_nodes(op="get_attr"):

            if node.target not in seen_targets:
                seen_targets.add(node.target)
                continue

            node.target = self._copy_attr(graph_module, node)
            modified = True

        if modified:
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
