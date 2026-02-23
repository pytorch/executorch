# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Callable, Iterator, List

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


class NodeFinder(ABC):
    @abstractmethod
    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        """Return nodes of the graph module depending on NodeFinder type.

        Args:
            model (GraphModule): The graph module to search for matching nodes.
        """
        pass


class GlobalNodeFinder(NodeFinder):
    """
    Finds all nodes of the graph.
    """

    def find_nodes(self, model: GraphModule) -> Iterator[Node]:
        return (n for n in model.graph.nodes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} targeting all nodes"


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
