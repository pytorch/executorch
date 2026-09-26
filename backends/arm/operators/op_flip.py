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
class FlipVisitor(NodeVisitor):
    target = "aten.flip.default"

    def __init__(self, *args):
        super().__init__(*args)

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
        if self.tosa_spec.support_extension("fp8e4m3"):
            supported_dtypes.append(ts.DType.FP8E4M3)
        if self.tosa_spec.support_extension("fp8e5m2"):
            supported_dtypes.append(ts.DType.FP8E5M2)

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            supported_dtypes,
            self.tosa_spec,
        )

        dims = inputs[1].special
        if len(dims) != 1:
            raise ValueError(
                f"{self.target} expects a single flip dim after DecomposeFlipPass, "
                f"got {dims}"
            )
        axis = dims[0] % len(inputs[0].shape)

        attr = ts.TosaSerializerAttribute()
        attr.ReverseAttribute(axis)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.REVERSE,
            [inputs[0].name],
            [output.name],
            attr,
        )
