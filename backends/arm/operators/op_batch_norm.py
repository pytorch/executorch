# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import serializer.tosa_serializer as ts
import torch
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_utils import promote_shape, tosa_shape
from serializer.tosa_serializer import TosaOp


@register_node_visitor
class BatchNormVisitor(NodeVisitor):
    target = "aten._native_batch_norm_legit_no_training.default"

    def __init__(self, *args):
        super().__init__(*args)

    # For BatchNorm2D, mean and var are calculated over the channel dimension
    # But TOSA doesn't allow subtraction of inputs with different ranks
    # Need to augment the shapes to match the ranks with activations
    def augment_shape_rank(self, shape, dim_order):
        nchw_shape = (1, *shape, 1, 1)
        return tosa_shape(nchw_shape, dim_order)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
        is_quant_node: bool,
    ) -> None:
        # Decompose batch norm into sequence
        (activations, weights, bias, running_mean, running_var, momentum, epsilon) = (
            inputs
        )

        input_dtype = activations.dtype

        assert (
            0.1 == momentum.number
        ), "Expected 0.1 momentum, not currently encoded into TOSA"

        # %output = (%x - %E[x]) /  SQRT( %Var[x] + %epsilon ) * %gamma + %beta
        # e.g.
        # %output = (%activations - %running_mean) /  SQRT( %running_var + %epsilon_const ) * %weights +  %bias
        # ->
        # %op1 = tosa.SUB(%activations, %running_mean)
        # %op2 = tosa.ADD(%running_var, %epsilon_const)
        # %op3 = tosa.RSQRT(%op2)
        # %op4 = tosa.MUL(%op1, %op3)
        # %op5 = tosa.MUL(%op4, %weights)
        # %output = tosa.ADD(%op5, %bias)

        # Reshape mean to match rank of activations
        mean_reshaped = promote_shape(
            tosa_graph,
            running_mean,
            self.augment_shape_rank(running_mean.shape, output.dim_order),
            input_dtype,
        )

        # Subtract mean
        # %op1 = tosa.SUB(%activations, %running_mean)
        op1 = tosa_graph.addIntermediate(
            tosa_shape(output.shape, output.dim_order), input_dtype
        )
        tosa_graph.addOperator(
            TosaOp.Op().SUB,
            [activations.name, mean_reshaped.name],
            [op1.name],
        )
        # Adding eplison to variance
        # %op2 = tosa.ADD(%running_var, %epsilon_const)
        epsilon_const = tosa_graph.addConst([1], input_dtype, [epsilon.number])
        op2 = tosa_graph.addIntermediate(
            tosa_shape(running_var.shape, running_var.dim_order), input_dtype
        )
        tosa_graph.addOperator(
            TosaOp.Op().ADD,
            [running_var.name, epsilon_const.name],
            [op2.name],
        )
        # Push downward the variance
        # %op3 = tosa.RSQRT(%op2)
        op3 = tosa_graph.addIntermediate(running_var.shape, input_dtype)
        tosa_graph.addOperator(TosaOp.Op().RSQRT, [op2.name], [op3.name])

        # Reshape variable to match rank of activations
        op3_reshaped = promote_shape(
            tosa_graph,
            op3,
            self.augment_shape_rank(running_var.shape, output.dim_order),
            input_dtype,
        )

        # Handle non existing weights and bias
        if not weights.name and not bias.name:
            # Multiply shifted activations with reciprocal variance
            # %output = tosa.MUL(%op1, %op3)  e.g. Now we have %output = (%activations - %running_mean) /  SQRT( %running_var + %epsilon_const )
            attr_mul = ts.TosaSerializerAttribute()
            attr_mul.MulAttribute(0)
            tosa_graph.addOperator(
                TosaOp.Op().MUL, [op1.name, op3_reshaped.name], [output.name], attr_mul
            )
            return
        else:
            # Multiply shifted activations with reciprocal variance
            # %op4 = tosa.MUL(%op1, %op3)
            op4 = tosa_graph.addIntermediate(
                tosa_shape(output.shape, output.dim_order), input_dtype
            )
            attr_mul = ts.TosaSerializerAttribute()
            attr_mul.MulAttribute(0)
            tosa_graph.addOperator(
                TosaOp.Op().MUL, [op1.name, op3_reshaped.name], [op4.name], attr_mul
            )

        # Now we have %op4 = (%activations - %running_mean) /  SQRT( %running_var + %epsilon_const )

        if weights.name and not bias.name:
            # Handle only weights but no bias

            # Reshape weights to match rank of activations
            weights_reshaped = promote_shape(
                tosa_graph,
                weights,
                self.augment_shape_rank(weights.shape, output.dim_order),
                input_dtype,
            )

            # %output = tosa.MUL(%op4, %weights)
            attr_mul = ts.TosaSerializerAttribute()
            attr_mul.MulAttribute(0)
            tosa_graph.addOperator(
                TosaOp.Op().MUL,
                [op4.name, weights_reshaped.name],
                [output.name],
                attr_mul,
            )
            return

        if not weights.name and bias.name:
            # Handle only bias but no weights

            # Reshape bias to match rank of activations
            bias_reshaped = promote_shape(
                tosa_graph,
                bias,
                self.augment_shape_rank(bias.shape, output.dim_order),
                input_dtype,
            )

            # %output = tosa.ADD(%op4, %bias)
            tosa_graph.addOperator(
                TosaOp.Op().ADD,
                [op4.name, bias_reshaped.name],
                [output.name],
            )
            return

        # We have both weights and bias

        # Reshape weights to match rank of activations
        weights_reshaped = promote_shape(
            tosa_graph,
            weights,
            self.augment_shape_rank(weights.shape, output.dim_order),
            input_dtype,
        )

        # %op5 = tosa.MUL(%op4, %weights)
        op5 = tosa_graph.addIntermediate(
            tosa_shape(output.shape, output.dim_order), input_dtype
        )
        attr_mul = ts.TosaSerializerAttribute()
        attr_mul.MulAttribute(0)
        tosa_graph.addOperator(
            TosaOp.Op().MUL,
            [op4.name, weights_reshaped.name],
            [op5.name],
            attr_mul,
        )

        # Reshape bias to match rank of activations
        bias_reshaped = promote_shape(
            tosa_graph,
            bias,
            self.augment_shape_rank(bias.shape, output.dim_order),
            input_dtype,
        )

        # %output = tosa.ADD(%op5, %bias)
        tosa_graph.addOperator(
            TosaOp.Op().ADD,
            [op5.name, bias_reshaped.name],
            [output.name],
        )
