# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch.fx
import tosa_serializer as ts

from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class ArgMaxVisitor(NodeVisitor):
    target = "tosa.ARGMAX.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        validate_valid_dtype(
            self.target,
            inputs[0],
            [
                ts.DType.INT8,
                ts.DType.INT16,
                ts.DType.FP16,
                ts.DType.FP32,
                ts.DType.BF16,
            ],
            self.tosa_spec,
        )
        validate_valid_dtype(self.target, output, ts.DType.INT32, self.tosa_spec)

        axis = inputs[1].number
        if axis < 0:
            tensor = get_first_fake_tensor(node)
            axis += len(tensor.size())

        attr = ts.TosaSerializerAttribute()
        attr.ArgMaxAttribute(axis, ts.NanPropagationMode.PROPAGATE)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.ARGMAX,
            [inputs[0].name],
            [output.name],
            attr,
        )
