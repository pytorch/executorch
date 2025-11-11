# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import fx
from torch.fx.passes.operator_support import OperatorSupportBase


def _compare(exact: bool, search_for: str | None, search_in: str) -> bool:
    """Check whether the search_for str matches the search_in str.
    Match can mean "identical" or "part of" depending on the `exact` flag.
    """
    if not search_for:
        return False
    if exact:
        return search_for == search_in
    else:
        return search_for in search_in


class DontSupportBase(OperatorSupportBase):
    _rejected_nodes: list[fx.Node] = []

    def reject_node(self, node: fx.Node):
        self._rejected_nodes.append(node)

    def rejected_nodes(self):
        return self._rejected_nodes

    def has_rejected_node(self) -> bool:
        return self.num_rejected() > 0

    def num_rejected(self) -> int:
        return len(self._rejected_nodes)


class DontPartition(DontSupportBase):
    """Operator check to skip partitioning ops based on their target.
    The target can be an EdgeOverloadOp (exir_ops.edge.aten.*),
    OverloadOp (torch.ops.aten.*), or a string ("aten.*").

    For the string case, set `exact` to False to match only part of the name.
    """

    def __init__(self, *targets, exact: bool = True):
        self.targets = targets
        self.exact = exact

    def is_node_supported(self, submodules, node: fx.Node) -> bool:
        if node.target in self.targets:
            self.reject_node(node)
            return False

        if "original_aten" not in node.meta:
            return True
        stringified_node_target = str(node.meta["original_aten"])
        for target in self.targets:
            if _compare(self.exact, str(target), stringified_node_target):
                self.reject_node(node)
                return False
        return True


class DontPartitionName(DontSupportBase):
    """Operator check to skip partitioning ops based on their name, which can be found
    by for example node.name or print-outs of a GraphModule.

    Set `exact` to False to match only part of the name.
    """

    def __init__(self, *targets, exact: bool = True):
        self.targets = targets
        self.exact = exact

    def is_node_supported(self, submodules, node: fx.Node) -> bool:
        for target in self.targets:
            if _compare(self.exact, target, node.name):
                self.reject_node(node)
                return False
        return True


class DontPartitionModule(DontSupportBase):
    """Operator check to skip partitioning modules.
    You can pass either the module name, i.e. the class name of the module,
    or the name of the instance that you want to skip.
    If module_name contains a dot, the full module name of checked nodes is used,
    if it does not, only part after the last dot is used.

    For example, you could have two files defining MyClass, which have the full module name:
        my_file.MyClass
        my_other_file.MyClass
    If you would call DontPartitionModule with module_name="MyClass", you would skip partitioning both.
    With "my_file.MyClass", you would only target the first class.

    Set `exact` to False to match only part of the name.
    """

    def __init__(
        self,
        *,
        module_name: str | None = None,
        instance_name: str | None = None,
        exact: bool = True,
    ):
        self.module_name = module_name
        self.instance_name = instance_name
        self.exact = exact
        self.used_dotted = "." in module_name if module_name else True

    def is_node_supported(self, submodules, node: fx.Node) -> bool:
        if "nn_module_stack" not in node.meta:
            return True

        for module_meta in node.meta["nn_module_stack"].values():
            if _compare(self.exact, self.instance_name, module_meta[0]):
                self.reject_node(node)
                return False
            node_module_name = module_meta[1]
            if not self.used_dotted:
                node_module_name = node_module_name.split(".")[-1]
            if _compare(self.exact, self.module_name, node_module_name):
                self.reject_node(node)
                return False

        return True
