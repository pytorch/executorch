# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, cast, List

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
class MinVisitor(NodeVisitor):
    target = "tosa.REDUCE_MIN.default"

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 1)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        valid_dtypes = [
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.FP16,
            ts.DType.FP32,
            ts.DType.BF16,
        ]
        if self.tosa_spec.is_U55_subset:
            valid_dtypes.remove(ts.DType.INT32)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            valid_dtypes,
            self.tosa_spec,
        )

        input = inputs[0]

        attr = ts.TosaSerializerAttribute()
        nan_mode = getattr(
            ts.NanPropagationMode, cast(str, node.kwargs.get("nan_mode", "PROPAGATE"))
        )
        attr.ReduceMinAttribute(axis=node.kwargs["axis"], nan_mode=nan_mode)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.REDUCE_MIN,
            [input.name],
            [output.name],
            attr,
        )
