# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import promote_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class BatchNormVisitor(NodeVisitor):
    target = "aten._native_batch_norm_legit_no_training.default"

    def __init__(self, *args):
        super().__init__(*args)

    # For BatchNorm2D, mean and var are calculated over the channel dimension
    # But TOSA doesn't allow subtraction of inputs with different ranks
    # Need to augment the shapes to match the ranks with activations
    def augment_shape_rank(self, input, permute_memory_to_nhwc):
        return (
            (1, 1, 1) + input.shape
            if permute_memory_to_nhwc
            else ((1,) + input.shape + (1, 1))
        )

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
        permute_memory_to_nhwc: bool,
    ) -> None:
        # Decompose batch norm into sequence
        (activations, _, _, running_mean, running_var, momentum, epsilon) = inputs

        input_dtype = activations.dtype

        assert (
            0.1 == momentum.number
        ), "Expected 0.1 momentum, not currently encoded into TOSA"

        # %op1 = tosa.SUB(%x, %bmean)
        # %op2 = tosa.ADD(%variance, %epsilon_const)
        # %op3 = tosa.RSQRT(%op2)
        # %op4 = tosa.MUL(%op1, %op3)
        # %op5 = tosa.MUL(%op4, %weight)
        # %output = tosa.ADD(%op5, %bias)

        # Reshape mean to match rank of activations
        mean_reshaped_res = promote_shape(
            tosa_graph,
            running_mean,
            self.augment_shape_rank(running_mean, permute_memory_to_nhwc),
            input_dtype,
        )

        # Subtract mean
        int1 = tosa_graph.addIntermediate(output.shape, input_dtype)
        tosa_graph.addOperator(
            TosaOp.Op().SUB,
            [activations.name, mean_reshaped_res.name],
            [int1.name],
        )
        # Adding eplison to variance
        epsilon_const = tosa_graph.addConst([1], input_dtype, [epsilon.number])
        int2 = tosa_graph.addIntermediate(running_var.shape, input_dtype)
        tosa_graph.addOperator(
            TosaOp.Op().ADD,
            [running_var.name, epsilon_const.name],
            [int2.name],
        )
        # Push downward the variance
        int3 = tosa_graph.addIntermediate(running_var.shape, input_dtype)
        tosa_graph.addOperator(TosaOp.Op().RSQRT, [int2.name], [int3.name])

        # Reshape variable to match rank of activations
        var_reshaped_res = promote_shape(
            tosa_graph,
            int3,
            self.augment_shape_rank(running_var, permute_memory_to_nhwc),
            input_dtype,
        )

        attr_mul = ts.TosaSerializerAttribute()
        attr_mul.MulAttribute(0)

        # Multiple shifted activations with reciprocal variance
        tosa_graph.addOperator(
            TosaOp.Op().MUL, [int1.name, var_reshaped_res.name], [output.name], attr_mul
        )
