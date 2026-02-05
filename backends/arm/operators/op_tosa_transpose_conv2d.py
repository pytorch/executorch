# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tosa_serializer as ts

"""Provide a visitor for lowering 2D transpose convolution to TOSA (INT/FP)."""
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_valid_dtype,
)
from executorch.backends.arm.operators.ops_quant_utils import add_input_weight_zp_consts
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_node_visitor
class TransposeConv2dVisitor(NodeVisitor):
    """Provide a visitor that serializes TOSA ``TRANSPOSE_CONV2D``."""

    target = "tosa.TRANSPOSE_CONV2D.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def _get_tosa_op(self):
        return ts.Op.TRANSPOSE_CONV2D

    def _get_attr_func(self, attr):
        return attr.TransposeConv2dAttribute

    def define_node(
        self,
        node,
        tosa_graph,
        inputs: list[TosaArg],
        output: TosaArg,
    ) -> None:
        """Define the TOSA TRANSPOSE_CONV2D operator."""

        input, weight, bias, out_pad, stride = inputs
        validate_num_inputs(self.target, inputs, 5)

        valid_input_dtypes = []
        if self.tosa_spec.support_float():
            valid_input_dtypes.append(ts.DType.FP32)
        if self.tosa_spec.support_integer():
            valid_input_dtypes.append(ts.DType.INT8)

        if self.tosa_spec.support_extension("int16"):
            valid_input_dtypes.append(ts.DType.INT16)
            if inputs[0].dtype == ts.DType.INT16:
                validate_valid_dtype(
                    self.target, [inputs[1]], [ts.DType.INT8], self.tosa_spec
                )
                validate_valid_dtype(
                    self.target, [inputs[2]], [ts.DType.INT48], self.tosa_spec
                )

        validate_valid_dtype(
            self.target,
            [inputs[0]],
            valid_input_dtypes,
            self.tosa_spec,
        )

        out_pad_attr = out_pad.special
        stride_attr = stride.special

        output_name = output.name
        acc_type = output.dtype
        input_zp_name, weight_zp_name = add_input_weight_zp_consts(
            tosa_graph, node, inputs, output_name
        )

        attr = ts.TosaSerializerAttribute()
        self._get_attr_func(attr)(
            out_pad=out_pad_attr,
            stride=stride_attr,
            local_bound=False,
            acc_type=acc_type,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            self._get_tosa_op(),
            [
                input.name,
                weight.name,
                bias.name,
                input_zp_name,
                weight_zp_name,
            ],
            [output_name],
            attr,
        )
