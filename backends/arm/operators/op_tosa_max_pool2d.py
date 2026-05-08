# Copyright 2024-2026 Arm Limited and/or its affiliates.
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
class MaxPool2dVisitor(NodeVisitor):
    """Visitor for lowering TOSA MAX_POOL2D operator."""

    target = "tosa.MAX_POOL2D.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, [4])
        validate_same_dtype(self.target, [inputs[0], output], ts)

        input_tensor, kernel, stride, pad = inputs

        supported_dtypes = [ts.DType.INT8, ts.DType.FP16, ts.DType.FP32, ts.DType.BF16]
        if self.tosa_spec.support_extension("int16"):
            supported_dtypes.append(ts.DType.INT16)
        validate_valid_dtype(
            self.target,
            [input_tensor, output],
            supported_dtypes,
            self.tosa_spec,
        )

        attr = ts.TosaSerializerAttribute()
        attr.MaxPool2dAttribute(
            kernel=kernel.special,
            stride=stride.special,
            pad=pad.special,
            nan_mode=ts.NanPropagationMode.PROPAGATE,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.MAX_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )
