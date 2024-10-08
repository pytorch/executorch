# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict, List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.tosa_mapping import TosaArg
from torch.export import ExportedProgram


class NodeVisitor:
    """
    Node Visitor pattern for lowering edge IR to TOSA
    """

    def __init__(self, exported_program: ExportedProgram):
        self._exported_program = exported_program or None

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        raise NotImplementedError("NodeVisitor must be extended.")


# container for all node visitors
_node_visitor_dict = {}


def register_node_visitor(visitor):
    _node_visitor_dict[visitor.target] = visitor


def get_node_visitors(*args) -> Dict[str, NodeVisitor]:
    node_visitors = {}
    for target, visitor in _node_visitor_dict.items():
        node_visitors[target] = visitor(*args)

    return node_visitors
