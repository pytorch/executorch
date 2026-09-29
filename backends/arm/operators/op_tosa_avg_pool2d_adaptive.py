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


if hasattr(ts.Op, "AVG_POOL2D_ADAPTIVE"):

    @register_node_visitor
    class AvgPool2dAdaptiveVisitor(NodeVisitor):
        """Visitor for lowering TOSA AVG_POOL2D_ADAPTIVE operator."""

        target = "tosa.AVG_POOL2D_ADAPTIVE.default"

        def define_node(
            self,
            node: torch.fx.Node,
            tosa_graph: Any,
            inputs: List[TosaArg],
            output: TosaArg,
        ) -> None:
            validate_num_inputs(self.target, inputs, [7])
            validate_same_dtype(self.target, [inputs[0], output], ts)

            input_tensor, input_zp, output_zp, kernel, stride, pad, acc_arg = inputs

            supported = [ts.DType.INT8, ts.DType.FP16, ts.DType.FP32, ts.DType.BF16]
            if self.tosa_spec.support_extension("int16"):
                supported.append(ts.DType.INT16)
            if self.tosa_spec.support_extension("fp8e4m3"):
                supported.append(ts.DType.FP8E4M3)
            if self.tosa_spec.support_extension("fp8e5m2"):
                supported.append(ts.DType.FP8E5M2)
            validate_valid_dtype(
                self.target, [input_tensor, output], supported, self.tosa_spec
            )

            attr = ts.TosaSerializerAttribute()
            attr.AvgPool2dAdaptiveAttribute(acc_type=acc_arg.dtype)

            self._serialize_operator(
                node,
                tosa_graph,
                ts.Op.AVG_POOL2D_ADAPTIVE,
                [
                    input_tensor.name,
                    input_zp.name,
                    output_zp.name,
                    kernel.name,
                    stride.name,
                    pad.name,
                ],
                [output.name],
                attr,
            )
