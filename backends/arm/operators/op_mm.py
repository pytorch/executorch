# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch

# pyre-fixme[21]: 'Could not find a module corresponding to import `executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass`.'
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_quant_utils import build_rescale
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.arm.tosa_utils import build_reshape, expand_dims
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class MMVisitor_080_BI(NodeVisitor):
    target = "aten.mm.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        # For atem.mm, the two inputs are of rank 2
        # For TOSA it needs to be rank 3
        # So they need to be reshaped from (H, W) to (1, H, W)
        reshape_dtype = output.dtype
        input0_reshaped = expand_dims(tosa_graph, inputs[0], reshape_dtype, 0)
        input1_reshaped = expand_dims(tosa_graph, inputs[1], reshape_dtype, 0)

        # The output also needs to be rank 3
        output_new_shape = (1, output.shape[0], output.shape[1])

        input_qparams = get_input_qparams(node)  # pyre-ignore[16]
        assert len(input_qparams) == 2
        input0_qparams = input_qparams[0]
        input1_qparams = input_qparams[1]

        mat_mul_result = tosa_graph.addIntermediate(output_new_shape, ts.DType.INT32)

        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(A_zp=input0_qparams.zp, B_zp=input1_qparams.zp)

        tosa_graph.addOperator(
            TosaOp.Op().MATMUL,
            [input0_reshaped.name, input1_reshaped.name],
            [mat_mul_result.name],
            attr,
        )

        reshape_intermediate = tosa_graph.addIntermediate(output.shape, ts.DType.INT32)
        reshape_output_name = reshape_intermediate.name

        # Reshape the final output back to rank 2
        build_reshape(
            tosa_graph, mat_mul_result.name, output.shape, reshape_output_name
        )

        # As INT8 accumulates into INT32, we need to rescale it back to INT8
        output_qparams = get_output_qparams(node)  # pyre-ignore[16]
        assert len(output_qparams) == 1

        final_output_scale = (
            input0_qparams.scale * input1_qparams.scale
        ) / output_qparams[0].scale

        # As the input will be INT32, the input_zp must be set to 0
        build_rescale(
            tosa_fb=tosa_graph,
            scale=final_output_scale,
            input_node=reshape_intermediate,
            output_name=output.name,
            output_type=output.dtype,
            output_shape=reshape_intermediate.shape,
            input_zp=0,
            output_zp=output_qparams[0].zp,
            is_double_round=False,
        )


@register_node_visitor
class MMVisitor_080_MI(MMVisitor_080_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        if inputs[0].dtype == ts.DType.INT8:
            return super().define_node(node, tosa_graph, inputs, output)
        reshape_dtype = output.dtype
        # For atem.mm, the two inputs are of rank 2
        # For TOSA it needs to be rank 3
        # So they need to be reshaped from (H, W) to (1, H, W)
        input0_reshaped = expand_dims(tosa_graph, inputs[0], reshape_dtype, 0)
        input1_reshaped = expand_dims(tosa_graph, inputs[1], reshape_dtype, 0)

        # The output also needs to be rank 3
        output_new_shape = (1, output.shape[0], output.shape[1])

        # Set zps to 0
        input0_zp, input1_zp = 0, 0
        attr = ts.TosaSerializerAttribute()
        attr.MatMulAttribute(A_zp=input0_zp, B_zp=input1_zp)
        mat_mul_result = tosa_graph.addIntermediate(output_new_shape, output.dtype)
        reshape_output_name = output.name

        tosa_graph.addOperator(
            TosaOp.Op().MATMUL,
            [input0_reshaped.name, input1_reshaped.name],
            [mat_mul_result.name],
            attr,
        )
        # Reshape the final output back to rank 2
        build_reshape(
            tosa_graph, mat_mul_result.name, output.shape, reshape_output_name
        )
