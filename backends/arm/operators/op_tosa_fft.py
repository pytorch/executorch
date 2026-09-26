# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch.fx
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
class FFT2dVisitor(NodeVisitor):
    target = "tosa.FFT2D.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, inputs, ts)
        validate_valid_dtype(self.target, inputs, ts.DType.FP32, self.tosa_spec)

        attr = ts.TosaSerializerAttribute()
        attr.FFT2dAttribute(
            node.kwargs.get("inverse", False),
            node.kwargs.get("local_bound", False),
        )
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.FFT2D,
            [inputs[0].name, inputs[1].name],
            output.multiple_output_names,
            attr,
        )


@register_node_visitor
class RFFT2dVisitor(NodeVisitor):
    target = "tosa.RFFT2D.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 1)
        validate_valid_dtype(self.target, inputs, ts.DType.FP32, self.tosa_spec)

        attr = ts.TosaSerializerAttribute()
        attr.RFFT2dAttribute(node.kwargs.get("local_bound", False))
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.RFFT2D,
            [inputs[0].name],
            output.multiple_output_names,
            attr,
        )
