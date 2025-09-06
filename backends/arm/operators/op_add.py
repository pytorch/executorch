# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import executorch.backends.arm.tosa.quant_utils as tqutils
import executorch.backends.arm.tosa.utils as tutils

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg
from torch.fx import Node


@register_node_visitor
class AddVisitor_INT(NodeVisitor):
    target = "aten.add.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)
        valid_dtypes = []
        if self.tosa_spec.support_integer():
            valid_dtypes.extend([ts.DType.INT8, ts.DType.INT16, ts.DType.INT32])
        if self.tosa_spec.support_float():
            valid_dtypes.extend([ts.DType.INT32])

        validate_valid_dtype(
            self.target,
            [*inputs, output],
            valid_dtypes,
            output.tosa_spec,
        )
        scale_back = 1.0
        if inputs[0].dtype == ts.DType.INT8:
            rescaled_inputs, scale_back = tqutils.insert_rescale_ops_to_int32_maxscale(
                tosa_graph, inputs, node, self.tosa_spec
            )
        else:
            # input[0].dtype == ts.DType.INT16 or ts.DType.INT32
            # Non quantized input, natively support by TOSA.ADD
            rescaled_inputs = inputs

        if output.dtype == ts.DType.INT8:
            broadcasted_shape = tutils.tosa_shape(output.shape, output.dim_order)
            add_output = tosa_graph.addIntermediate(broadcasted_shape, ts.DType.INT32)
        else:
            # output.dtype == ts.DType.INT16 or ts.DType.INT32
            add_output = output

        input1, input2 = rescaled_inputs

        # Do the INT32 Add
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().ADD,
            [input1.name, input2.name],
            [add_output.name],
            None,
        )

        if output.dtype == ts.DType.INT8:
            # Scale output back to 8 bit
            # pyre-ignore
            tqutils.insert_rescale_op_to_int8(
                tosa_graph,
                add_output,
                scale_back,
                node,
                compute_rescale=False,
                tosa_spec=self.tosa_spec,
            )  # type: ignore[possibly-undefined]


@register_node_visitor
class AddVisitor_FP(AddVisitor_INT):
    # inheriting 'target' from INT class

    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.0+FP")]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)

        if inputs[0].dtype in [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32]:
            # Call the inherited define_node for handling integers
            super().define_node(node, tosa_graph, inputs, output)
        else:
            # FP32 Add lowering
            validate_valid_dtype(
                self.target, [*inputs, output], ts.DType.FP32, output.tosa_spec
            )

            input1, input2 = inputs

            # FP lowering
            self._serialize_operator(
                node,
                tosa_graph,
                ts.TosaOp.Op().ADD,
                [input1.name, input2.name],
                [output.name],
                None,
            )
