# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict, List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from torch.export import ExportedProgram


class NodeVisitor:
    """
    Node Visitor pattern for lowering edge IR to TOSA
    """

    # Add the currently supported node_visitor specs as default.
    # This should be overriden in the NodeVisitor subclasses to target
    # a specific TOSA version.
    # When all node_visitors has been refactored to target a specific
    # version, this list should be removed.
    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80.0+BI"),
        TosaSpecification.create_from_string("TOSA-0.80.0+MI"),
    ]

    def __init__(self, exported_program: ExportedProgram, tosa_spec: TosaSpecification):
        self._exported_program = exported_program or None
        self.tosa_spec = tosa_spec

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
_node_visitor_dicts = {
    TosaSpecification.create_from_string("TOSA-0.80.0+BI"): {},
    TosaSpecification.create_from_string("TOSA-0.80.0+MI"): {},
}


def register_node_visitor(visitor):
    for tosa_spec in visitor.tosa_specs:
        _node_visitor_dicts[tosa_spec][visitor.target] = visitor
    return visitor


def get_node_visitors(*args) -> Dict[str, NodeVisitor]:
    node_visitors = {}
    tosa_spec = None
    for arg in args:
        if isinstance(arg, TosaSpecification):
            tosa_spec = arg
            break

    if tosa_spec is None:
        raise RuntimeError("No TOSA specification supplied.")

    for target, visitor in _node_visitor_dicts[tosa_spec].items():
        node_visitors[target] = visitor(*args)

    return node_visitors
