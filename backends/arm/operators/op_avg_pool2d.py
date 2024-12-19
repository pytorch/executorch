# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch

# pyre-fixme[21]: ' Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification


@register_node_visitor
class AvgPool2dVisitor_0_80_BI(NodeVisitor):
    target = "aten.avg_pool2d.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def _build_generic_avgpool2d(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        input_zp: int,
        output_zp: int,
        accumulator_type,
    ) -> None:
        input_tensor = inputs[0]

        kernel_size_list = inputs[1].special
        stride_size_list = inputs[2].special
        try:
            pad_size_list = inputs[3].special
        except IndexError:
            pad_size_list = [0, 0, 0, 0]

        attr = ts.TosaSerializerAttribute()
        attr.PoolAttribute(
            kernel=kernel_size_list,
            stride=stride_size_list,
            pad=pad_size_list,
            input_zp=input_zp,
            output_zp=output_zp,
            accum_dtype=accumulator_type,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().AVG_POOL2D,
            [input_tensor.name],
            [output.name],
            attr,
        )

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        input_tensor = inputs[0]
        assert input_tensor.dtype == ts.DType.INT8

        accumulator_type = ts.DType.INT32

        input_qargs = get_input_qparams(node)  # pyre-ignore[16]
        input_zp = input_qargs[0].zp

        output_qargs = get_output_qparams(node)  # pyre-ignore[16]
        output_zp = output_qargs[0].zp

        self._build_generic_avgpool2d(
            node, tosa_graph, inputs, output, input_zp, output_zp, accumulator_type
        )


@register_node_visitor
class AvgPool2dVisitor_0_80_MI(AvgPool2dVisitor_0_80_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        assert (
            inputs[0].dtype == ts.DType.INT8 or inputs[0].dtype == ts.DType.FP32
        ), "Only FP32 and INT8 supported"

        if inputs[0].dtype == ts.DType.INT8:
            super().define_node(node, tosa_graph, inputs, output, is_quant_node)

        if inputs[0].dtype == ts.DType.FP32:
            accumulator_type = ts.DType.FP32
            # Initilize zero point to zero.
            input_zp = 0
            output_zp = 0

            self._build_generic_avgpool2d(
                node, tosa_graph, inputs, output, input_zp, output_zp, accumulator_type
            )
