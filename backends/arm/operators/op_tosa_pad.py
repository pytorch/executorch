# Copyright 2026 Arm Limited and/or its affiliates.
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
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class TosaPadVisitor(NodeVisitor):
    target = "tosa.PAD.default"

    tosa_specs = NodeVisitor.tosa_specs

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        supported_dtypes = [ts.DType.BOOL]
        if self.tosa_spec.support_integer():
            supported_dtypes.extend([ts.DType.INT8, ts.DType.INT16, ts.DType.INT32])
        if self.tosa_spec.support_float():
            supported_dtypes.extend([ts.DType.FP16, ts.DType.FP32])
        if self.tosa_spec.support_extension("bf16"):
            supported_dtypes.append(ts.DType.BF16)

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            supported_dtypes,
            self.tosa_spec,
        )

        pad_const = tosa_graph.addConst(
            [1],
            output.dtype,
            [node.kwargs.get("value", 0)],
            name=node.name + "_padding_value",
        )

        attr = ts.TosaSerializerAttribute()
        attr.PadAttribute()

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.PAD,
            [
                inputs[0].name,
                inputs[1].name,
                pad_const.name,
            ],
            [output.name],
            attr,
        )
