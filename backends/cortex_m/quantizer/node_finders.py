# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Iterator, List

import torch
from executorch.backends.arm.quantizer.arm_quantizer_utils import NodeFinder
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer.utils import get_module_name_filter


def make_list(item_or_list):
    if isinstance(item_or_list, (list, tuple, set)):
        return item_or_list
    else:
        return [item_or_list]


def format_items(items) -> str:
    """Render an iterable as a comma-separated string."""
    return ", ".join(str(item) for item in items)


class GlobalNodeFinder(NodeFinder):
    """
    Finds all nodes of the graph.
    """

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting all nodes"


class GlobalNoOutputNodeFinder(NodeFinder):
    """Finds all graph nodes except the output node. Output nodes
    typically wants to inherit the previous qspec rather than applying a new
    qspec."""

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes if n.op != "output")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting all non-output nodes"


class InputNodeFinder(NodeFinder):
    """
    Finds all placeholder nodes.
    """

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes if n.op == "placeholder")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting all placeholder nodes"


class OutputNodeFinder(NodeFinder):
    """
    Finds the output node.
    """

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes if n.op == "output")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting the output node"


class NodeNameNodeFinder(NodeFinder):
    """
    Finds all nodes matching the given name(s).
    """

    def __init__(self, names: str | List[str]) -> None:
        super().__init__()
        self.names = make_list(names)

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes if n.name in self.names)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting names: {format_items(self.names)}"


class NodeTargetNodeFinder(NodeFinder):
    """
    Finds all nodes matching the given target(s).
    """

    def __init__(self, targets: OpOverload | List[OpOverload]) -> None:
        super().__init__()
        self.targets = make_list(targets)

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes if n.target in self.targets)

    def __repr__(self) -> str:
        target_names = [t._name for t in self.targets]
        return f"{self.__class__.__name__} targeting node targets: {format_items(target_names)}"


class NodeTargetActivationNodeFinder(NodeFinder):
    """
    Finds all nodes matching the given target, and the nodes matching the given
    activation(s) that are the sole user of one of them.
    """

    def __init__(
        self, target: OpOverload, activations: OpOverload | List[OpOverload]
    ) -> None:
        super().__init__()
        self.target = target
        self.activations = make_list(activations)

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (
            n
            for n in model.graph.nodes
            if n.target == self.target
            or (
                n.target in self.activations
                and any(
                    i.target == self.target and len(i.users) == 1
                    for i in n.all_input_nodes
                )
            )
        )

    def __repr__(self) -> str:
        activation_names = [a._name for a in self.activations]
        return (
            f"{self.__class__.__name__} targeting node target: {self.target._name} "
            f"with activations: {format_items(activation_names)}"
        )


class ModuleNameNodeFinder(NodeFinder):
    """
    Finds all nodes in the module matching the given name(s).
    See arm_quantizer for original implementation.
    """

    def __init__(self, module_names: str | List[str]) -> None:
        super().__init__()
        self.module_names = make_list(module_names)
        module_name_filters = [
            get_module_name_filter(name) for name in self.module_names
        ]
        self.module_name_filter = lambda node: any(
            module_name_filter(node) for module_name_filter in module_name_filters
        )

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes if self.module_name_filter(n))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting module names: {format_items(self.module_names)}"


class ModuleTypeNodeFinder(NodeFinder):
    """
    Finds all nodes in the module matching the given type(s).
    See arm_quantizer for original implementation.
    """

    def _get_module_type_filter(self, tp: Callable) -> Callable:
        tp_str = tp.__module__ + "." + tp.__qualname__

        def module_type_filter(n: Node) -> bool:
            """Return True if the node originates from the target module type."""
            # node_stack example: {
            #     'L__self___sub': ("L['self'].sub", <class '....Sub'>),
            #     'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
            # }
            nn_module_stack = n.meta.get("nn_module_stack", {})
            types = [t for _, t in nn_module_stack.values()]
            return tp_str in types

        return module_type_filter

    def __init__(self, module_types: Callable | List[Callable]) -> None:
        super().__init__()
        module_types = make_list(module_types)
        self.module_type_names = [m.__name__ for m in module_types]

        module_type_filters = [self._get_module_type_filter(tp) for tp in module_types]
        self.module_type_filter = lambda node: any(
            module_type_filter(node) for module_type_filter in module_type_filters
        )

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes if self.module_type_filter(n))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting module types: {format_items(self.module_type_names)}"


class SDPAFinder(NodeFinder):
    """Find native SDPA calls and their FX decomposition descendants.

    Decompositions must preserve the original SDPA call in the standard
    ``source_fn_stack`` metadata. No backend-specific role tags are required.

    """

    @staticmethod
    def _groups(model: GraphModule) -> dict[str, set[Node]]:
        groups: dict[str, set[Node]] = {}
        for node in model.graph.nodes:
            if node.target == torch.ops.aten.scaled_dot_product_attention.default:
                groups.setdefault(node.name, set()).add(node)
                continue
            for name, target in reversed(node.meta.get("source_fn_stack", [])):
                if target in (
                    torch.ops.aten.scaled_dot_product_attention.default,
                    torch.nn.functional.scaled_dot_product_attention,
                ):
                    groups.setdefault(name, set()).add(node)
                    break
        return groups

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        nodes = set().union(*self._groups(model).values())
        return (node for node in model.graph.nodes if node in nodes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting SDPA calls and decompositions"


class SDPAMatmulFinder(SDPAFinder):
    """Find matmuls within SDPA, optionally selecting the QK or AV product.

    Args:
        role (str | None): ``qk`` selects the score product, ``av`` selects
            the value product, and ``None`` selects both. Products are
            classified by their dependency order within the SDPA call.

    """

    def __init__(self, role: str | None = None) -> None:
        if role not in (None, "qk", "av"):
            raise ValueError("SDPA matmul role must be qk, av, or None")
        self.role = role

    @staticmethod
    def _ancestors(node: Node, group: set[Node]) -> set[Node]:
        found: set[Node] = set()
        pending = list(node.all_input_nodes)
        while pending:
            upstream = pending.pop()
            if upstream not in found and (
                upstream in group
                or str(upstream.target).startswith("quantized_decomposed.")
            ):
                found.add(upstream)
                pending.extend(upstream.all_input_nodes)
        return found

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        selected: set[Node] = set()
        for group in self._groups(model).values():
            matmuls = {
                node
                for node in group
                if node.target
                in (torch.ops.aten.bmm.default, torch.ops.aten.matmul.default)
            }
            if self.role is None:
                selected.update(matmuls)
                continue
            if any(
                node.target == torch.ops.aten.scaled_dot_product_attention.default
                for node in group
            ):
                continue
            if len(matmuls) != 2:
                raise ValueError("Cannot identify QK and AV in SDPA decomposition")
            scores = set().union(
                *(matmuls & self._ancestors(node, group) for node in matmuls)
            )
            values = {node for node in matmuls if scores & self._ancestors(node, group)}
            if len(scores) != 1 or len(values) != 1 or scores == values:
                raise ValueError("Cannot identify QK and AV in SDPA decomposition")
            selected.update(scores if self.role == "qk" else values)
        return (node for node in model.graph.nodes if node in selected)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting SDPA matmuls: {self.role or 'all'}"


class SDPAScoreFinder(SDPAFinder):
    """Find SDPA score scaling, masking, and softmax intermediates.

    Keep the QK operands and matmul outside this region so their
    quantization can be configured independently of softmax precision.

    """

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        qk_products = set(SDPAMatmulFinder("qk").find_nodes(model))
        softmax_targets = (
            torch.ops.aten._safe_softmax.default,
            torch.ops.aten._softmax.default,
            torch.ops.aten.softmax.int,
        )
        selected: set[Node] = set()
        for group in self._groups(model).values():
            qk = qk_products & group
            softmax_nodes = {
                node
                for node in group
                if node.target in softmax_targets
                or any(
                    target in softmax_targets
                    for _, target in node.meta.get("source_fn_stack", [])
                )
            }
            score_nodes = softmax_nodes | set().union(
                *(SDPAMatmulFinder._ancestors(node, group) for node in softmax_nodes)
            )
            operands = set().union(
                *(SDPAMatmulFinder._ancestors(node, group) for node in qk)
            )
            selected.update((score_nodes & group) - operands - qk)
        return (node for node in model.graph.nodes if node in selected)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting SDPA scores and softmax"
