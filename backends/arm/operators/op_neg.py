# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch.fx

import tosa_serializer as ts

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
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


def get_negate_zero_points(node: torch.fx.Node, is_int8: bool) -> tuple[int, int]:
    """
    Returns (input1_zp, output_zp) for TOSA NEGATE.
    Must be zero for non-int8 types.
    """
    if is_int8:
        return (
            get_input_qparams(node)[0].get_zp_per_tensor(),
            get_output_qparams(node)[0].get_zp_per_tensor(),
        )
    return (0, 0)


@register_node_visitor
class NegVisitor(NodeVisitor):
    target = "aten.neg.default"

    tosa_specs = NodeVisitor.tosa_specs

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        supported_dtypes = [
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.FP16,
            ts.DType.BF16,
            ts.DType.FP32,
        ]

        validate_num_inputs(self.target, inputs, 1)
        validate_same_dtype(self.target, [*inputs, output], ts)
        validate_valid_dtype(
            self.target, [*inputs, output], supported_dtypes, output.tosa_spec
        )

        input_zp, output_zp = get_negate_zero_points(
            node, inputs[0].dtype == ts.DType.INT8
        )

        input_zp_tensor = tosa_graph.addConst(
            (1,), inputs[0].dtype, [input_zp], name=output.name + "_input_zp"
        )

        output_zp_tensor = tosa_graph.addConst(
            (1,), output.dtype, [output_zp], name=output.name + "_output_zp"
        )
        attr = ts.TosaSerializerAttribute()
        attr.NegateAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.NEGATE,
            [inputs[0].name, input_zp_tensor.name, output_zp_tensor.name],
            [output.name],
            attr,
        )
