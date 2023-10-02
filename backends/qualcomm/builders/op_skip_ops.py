# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)


class OpSkipOps(NodeVisitor):
    """
    Parent Class for handling Skip Ops
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> None:
        return


@register_node_visitor
class OpGetItem(OpSkipOps):
    """
    do nothing if node is getitem
    """

    target = "getitem"

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> None:
        if isinstance(node.args[1], tuple) or isinstance(node.args[1], list):
            raise AssertionError(
                f"Invalid number of index for {node.name }: {len(node.args[1])}"
            )
        nodes_to_wrappers[node] = nodes_to_wrappers.get(node.args[0])
        return
