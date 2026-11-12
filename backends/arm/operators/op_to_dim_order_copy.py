# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class ToDimOrderCopyVisitor(NodeVisitor):
    """Implement the type cast functionality of _to_dim_order_copy.

    Other features like setting of the dim_order or moving a tensor to a
    different device are not supported.

    Also note that the node should not be quantized.

    """

    target = "dim_order_ops._to_dim_order_copy.default"

    # TOSA casts between int64 and int8 or int16 through int32.
    _via_int32 = (
        {ts.DType.INT8, ts.DType.INT64},
        {ts.DType.INT16, ts.DType.INT64},
    )

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 1)
        source = inputs[0].name
        if {inputs[0].dtype, output.dtype} in self._via_int32:
            intermediate = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
            self._serialize_cast(node, tosa_graph, source, intermediate.name)
            source = intermediate.name
        self._serialize_cast(node, tosa_graph, source, output.name)

    def _serialize_cast(
        self, node: torch.fx.Node, tosa_graph: Any, source: str, target: str
    ) -> None:
        attr = ts.TosaSerializerAttribute()
        attr.CastAttribute()
        self._serialize_operator(node, tosa_graph, ts.Op.CAST, [source], [target], attr)
