# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import executorch.backends.arm.tosa.quant_utils as tqutils
import executorch.backends.arm.tosa.utils as tutils
import torch

from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
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
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.mapping import TosaArg


@register_node_visitor
class MulVisitor_INT(NodeVisitor):
    target = "aten.mul.Tensor"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+INT+int16"),
    ]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)
        validate_valid_dtype(
            self.target,
            [*inputs, output],
            [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32],
            output.tosa_spec,
        )

        if inputs[0].dtype == ts.DType.INT8 or inputs[0].dtype == ts.DType.INT16:
            input_A = inputs[0]
            input_B = inputs[1]
            input_qparams = get_input_qparams(node)
            input_A_qargs = input_qparams[0]
            input_B_qargs = input_qparams[1]
            input_A.shape = tutils.tosa_shape(input_A.shape, input_A.dim_order)
            input_B.shape = tutils.tosa_shape(input_B.shape, input_B.dim_order)

            # Rescale inputs to INT32 with zp=0
            input_A_rescaled = tqutils.build_rescale_to_int32(
                tosa_graph,
                input_A,
                input_A_qargs.get_zp_per_tensor(),
                1.0,
                tosa_spec=self.tosa_spec,
            )
            input_B_rescaled = tqutils.build_rescale_to_int32(
                tosa_graph,
                input_B,
                input_B_qargs.get_zp_per_tensor(),
                1.0,
                tosa_spec=self.tosa_spec,
            )
        else:
            # input[0].dtype == ts.DType.INT16 or ts.DType.INT32
            # Non quantized input, natively support by TOSA.MUL
            input_A_rescaled, input_B_rescaled = inputs[0], inputs[1]

        if output.dtype == ts.DType.INT8 or output.dtype == ts.DType.INT16:
            output_shape = tutils.tosa_shape(output.shape, output.dim_order)
            mul_output = tosa_graph.addIntermediate(output_shape, ts.DType.INT32)
        else:
            # output.dtype == ts.DType.INT32 (non-quantized)
            mul_output = output

        # Do the INT32 Mul
        tosa_graph.addConst([1], ts.DType.INT8, 0, name=f"{node.name}_shift")
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().MUL,
            [input_A_rescaled.name, input_B_rescaled.name, f"{node.name}_shift"],
            [mul_output.name],
        )

        if output.dtype == ts.DType.INT8:
            # Scale output back to 8 bit
            output_scale = (
                input_A_qargs.get_scale_per_tensor()  # type: ignore[possibly-undefined]
                * input_B_qargs.get_scale_per_tensor()  # type: ignore[possibly-undefined]
            )
            tqutils.insert_rescale_op_to_int8(
                tosa_graph, mul_output, output_scale, node, self.tosa_spec
            )
        elif output.dtype == ts.DType.INT16:
            # Scale output back to 16 bit
            output_scale = (
                input_A_qargs.get_scale_per_tensor()  # type: ignore[possibly-undefined]
                * input_B_qargs.get_scale_per_tensor()  # type: ignore[possibly-undefined]
            )
            tqutils.insert_rescale_op_to_int16(
                tosa_graph, mul_output, output_scale, node, self.tosa_spec
            )


@register_node_visitor
class MulVisitor_FP(MulVisitor_INT):
    # inheriting 'target' from INT class

    tosa_specs = [TosaSpecification.create_from_string("TOSA-1.0+FP")]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [*inputs, output], ts)

        if inputs[0].dtype == ts.DType.INT8:
            return super().define_node(node, tosa_graph, inputs, output)

        input1, input2 = inputs

        tosa_graph.addConst([1], ts.DType.INT8, 0, name=f"{node.name}_shift")
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().MUL,
            [input1.name, input2.name, f"{node.name}_shift"],
            [output.name],
        )
