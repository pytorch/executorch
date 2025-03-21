# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import numpy as np

import serializer.tosa_serializer as ts  # type: ignore
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class TableVisitor(NodeVisitor):
    target = "_table.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        if node.name not in self._exported_program.state_dict.keys():  # type: ignore[union-attr]
            raise RuntimeError(
                f"Did not find key {node.name} in state_dict {self._exported_program.state_dict.keys()}."
            )
        if inputs[0].dtype == ts.DType.INT8 and output.dtype != ts.DType.INT8:
            raise ValueError(f"Int8 tables need int8 output, got {output.dtype=}.")
        if inputs[0].dtype == ts.DType.INT16 and output.dtype != ts.DType.INT32:
            raise ValueError(f"Int16 tables need int32 output, got {output.dtype=}.")

        if inputs[0].dtype not in (ts.DType.INT8, ts.DType.INT16):
            raise ValueError(
                f"TOSA.TABLE only supports int8 or int16 inputs, got {ts.DTypeNames[inputs[0]]}"
            )

        table = self._exported_program.state_dict[node.name]  # type: ignore[union-attr]
        table_attr = ts.TosaSerializerAttribute()
        table_attr.TableAttribute(np.array(table))

        tosa_graph.addOperator(
            TosaOp.Op().TABLE, [inputs[0].name], [output.name], table_attr
        )
