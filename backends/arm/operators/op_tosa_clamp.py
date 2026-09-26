# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

from typing import Any, cast, List

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
from torch.fx import Node


@register_node_visitor
class ClampVisitor(NodeVisitor):
    target = "tosa.CLAMP.default"

    def _to_bytes(self, value: int | float, dtype: torch.dtype) -> List[int]:
        return cast(
            List[int],
            torch.full((1,), value, dtype=dtype).view(torch.uint8).numpy().tolist(),
        )

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, [2, 3])
        validate_same_dtype(self.target, [inputs[0], output], ts)
        supported_dtypes = [ts.DType.INT8, ts.DType.FP16, ts.DType.BF16, ts.DType.FP32]
        if self.tosa_spec.support_extension("int16"):
            supported_dtypes.append(ts.DType.INT16)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            supported_dtypes,
            self.tosa_spec,
        )

        node_input_dtype = node.meta["val"].dtype
        min_val = cast(int | float, node.args[1])
        max_val = cast(int | float, node.args[2])

        attr = ts.TosaSerializerAttribute()
        attr.ClampAttribute(
            self._to_bytes(min_val, node_input_dtype),
            self._to_bytes(max_val, node_input_dtype),
            nan_mode=ts.NanPropagationMode.PROPAGATE,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CLAMP,
            [inputs[0].name],
            [output.name],
            attr,
        )
