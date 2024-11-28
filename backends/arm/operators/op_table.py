# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import numpy as np

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class TableVisitor(NodeVisitor):
    target = "_table"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        assert node.name in self._exported_program.state_dict.keys()
        assert inputs[0].dtype == output.dtype == ts.DType.INT8
        table = self._exported_program.state_dict[node.name]
        table_attr = ts.TosaSerializerAttribute()
        table_attr.TableAttribute(np.array(table))
        tosa_graph.addOperator(
            TosaOp.Op().TABLE, [inputs[0].name], [output.name], table_attr
        )
