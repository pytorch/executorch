# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch
import torch.fx

import tosa_tools.v0_80.serializer.tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg


def identity_operator_factory(identity_target: str):
    """
    Creates and registers NodeVisitors for operators that map directly
    to a TOSA IDENTITY op.
    """

    class IdentityOperatorVisitor(NodeVisitor):
        target = identity_target

        def define_node(
            self,
            node: torch.fx.Node,
            tosa_graph: ts.TosaSerializer,
            inputs: List[TosaArg],
            output: TosaArg,
        ) -> None:
            # Simply add an identityOp
            tosa_graph.addOperator(
                ts.TosaOp.Op().IDENTITY, [inputs[0].name], [output.name]
            )

    register_node_visitor(IdentityOperatorVisitor)


identity_operator_factory("getitem")
identity_operator_factory("aten.alias_copy.default")
