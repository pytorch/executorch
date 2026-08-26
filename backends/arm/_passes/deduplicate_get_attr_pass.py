# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


class DeduplicateGetAttrPass(ArmPass):
    """Give duplicate get_attr nodes distinct backing attributes.

    Torchao's constant folder can delete a shared backing attribute while
    another get_attr node still refers to it. Keep separate graph nodes so PT2E
    can attach per-use observers and backend lowering can process constants per
    use.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def _replace_input_node(self, node: Node, old_node: Node, new_node: Node) -> None:
        def maybe_replace_node(arg: Any) -> Any:
            return new_node if arg is old_node else arg

        node.args = map_arg(node.args, maybe_replace_node)
        node.kwargs = map_arg(node.kwargs, maybe_replace_node)

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

    def _split_shared_get_attrs(self, graph_module: GraphModule) -> bool:
        modified = False

        for node in list(graph_module.graph.find_nodes(op="get_attr")):
            users = list(node.users)
            if len(users) <= 1:
                continue

            for user in users[1:]:
                with graph_module.graph.inserting_before(user):
                    new_node = graph_module.graph.get_attr(node.target)
                    new_node.meta.update(node.meta)
                self._replace_input_node(user, node, new_node)
                modified = True

        return modified

    def call(self, graph_module: GraphModule) -> PassResult:
        seen_targets: set[str] = set()
        modified = self._split_shared_get_attrs(graph_module)

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
