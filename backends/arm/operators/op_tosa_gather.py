# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class GatherVisitor(NodeVisitor):
    """
    Lowers backend TOSA dialect `tosa.GATHER.default`.

    Expected signature (per TOSA):
      values:  [N, K, C]  (rank 3)
      indices: [N, W]     (rank 2, int32)
      output:  [N, W, C]  (rank 3)
    """

    target = "tosa.GATHER.default"
    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)

        values = inputs[0]
        indices = inputs[1]

        validate_same_dtype(self.target, [values, output], ts)
        # Indices must be int32 for TOSA GATHER
        validate_valid_dtype(
            self.target,
            [indices],
            [ts.DType.INT32],
            self.tosa_spec,
        )
        validate_valid_dtype(
            self.target,
            [values, output],
            [
                ts.DType.INT8,
                ts.DType.INT16,
                ts.DType.INT32,
                ts.DType.FP16,
                ts.DType.FP32,
            ],
            self.tosa_spec,
        )

        attr = ts.TosaSerializerAttribute()
        attr.GatherAttribute()

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.GATHER,
            [values.name, indices.name],
            [output.name],
            attr,
        )
