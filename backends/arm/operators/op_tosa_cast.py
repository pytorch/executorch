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
from executorch.backends.arm.operators.operator_validation_utils import (  # type: ignore[attr-defined]
    supported_data_layout_dtypes,
    validate_num_inputs,
)
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class CastVisitor(NodeVisitor):
    target = "tosa.CAST.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, 2)
        supported_dtypes = supported_data_layout_dtypes(self.tosa_spec)
        for tensor in (inputs[0], output):
            if tensor.dtype not in supported_dtypes:
                valid_names = [str(dtype) for dtype in supported_dtypes]
                raise ValueError(
                    f"Expected tensor {tensor.name} in {self.target} to have one "
                    f"of the following dtypes: {valid_names}, got: {tensor.dtype}"
                )

        attr = ts.TosaSerializerAttribute()
        attr.CastAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CAST,
            [inputs[0].name],
            [output.name],
            attr,
        )
