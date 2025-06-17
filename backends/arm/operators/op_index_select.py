# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import executorch.backends.arm.tosa_quant_utils as tqutils  # noqa: F401

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg

from executorch.backends.arm.tosa_utils import build_reshape, build_reshape_tosa_1_0
from torch.fx import Node


@register_node_visitor
class IndexSelectVisitor(NodeVisitor):
    """
    Simple example:
          o = index_select(weights, index, indices)
    Becomes:
          i = view_copy(i)  # reshape flattened indicies, i.e. [I] => [1, I]
          o = index_select(w, index, i)

    Additional steps in case weights (w) are rank 2:
          - before: insert view_copy to make rank 3, [x,y] => [1, x, y]
          - after: insert view_copy to squeeze back output dims, [1, x, y] = [x,y]
    """

    target = "aten.index_select.default"
    tosa_specs = NodeVisitor.tosa_specs_1_00

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

        if len(inputs) != 3:
            raise ValueError(f"Number of inputs are not 3: {len(inputs)}")

        weights, index, indices = inputs

        if len(weights.shape) == 2:
            weights_new_shape = [1, weights.shape[0], weights.shape[1]]
            weights_reshaped = tosa_graph.addIntermediate(
                weights_new_shape,
                weights.dtype,
            )
            build_reshape_tosa_1_0(
                tosa_graph, weights.name, weights_new_shape, weights_reshaped.name
            )

            output_new_shape = [1, output.shape[0], output.shape[1]]
            output_reshaped = tosa_graph.addIntermediate(
                output_new_shape,
                output.dtype,
            )

        else:
            weights_reshaped = weights
            output_reshaped = output

        output_name = output_reshaped.name

        # Reshape flattened indicies, i.e. [I] => [1, I]
        indices_new_shape = [1, indices.shape[0]]
        indices_reshaped = tosa_graph.addIntermediate(
            indices_new_shape,
            indices.dtype,
        )
        build_reshape_tosa_1_0(
            tosa_graph, indices.name, indices_new_shape, indices_reshaped.name
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().GATHER,
            [weights_reshaped.name, indices_reshaped.name],
            [output_name],
            None,
        )

        if len(weights.shape) == 2:
            output_real_shape = [output.shape[0], output.shape[1]]
            build_reshape_tosa_1_0(
                tosa_graph, output_name, output_real_shape, output.name
            )


@register_node_visitor
class IndexSelectVisitor_0_80(NodeVisitor):
    """
    Simple example:
          o = index_select(weights, index, indices)
    Becomes:
          i = view_copy(i)  # reshape flattened indicies, i.e. [I] => [1, I]
          o = index_select(w, index, i)

    Additional steps in case weights (w) are rank 2:
          - before: insert view_copy to make rank 3, [x,y] => [1, x, y]
          - after: insert view_copy to squeeze back output dims, [1, x, y] = [x,y]
    """

    target = "aten.index_select.default"
    tosa_specs = NodeVisitor.tosa_specs_0_80

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts_v0_80  # type: ignore

        # Specification (0.80) states that input and output types
        # should all be the same
        if inputs[0].dtype != output.dtype:
            raise ValueError(
                f"Input and output type not same: {inputs[0].dtype} != {output.dtype:}"
            )

        if len(inputs) != 3:
            raise ValueError(f"Number of inputs are not 3: {len(inputs)}")

        weights, index, indices = inputs

        if len(weights.shape) == 2:
            weights_new_shape = [1, weights.shape[0], weights.shape[1]]
            weights_reshaped = tosa_graph.addIntermediate(
                weights_new_shape,
                weights.dtype,
            )
            build_reshape(
                tosa_graph, weights.name, weights_new_shape, weights_reshaped.name
            )

            output_new_shape = [1, output.shape[0], output.shape[1]]
            output_reshaped = tosa_graph.addIntermediate(
                output_new_shape,
                output.dtype,
            )

        else:
            weights_reshaped = weights
            output_reshaped = output

        output_name = output_reshaped.name

        # Reshape flattened indicies, i.e. [I] => [1, I]
        indices_new_shape = [1, indices.shape[0]]
        indices_reshaped = tosa_graph.addIntermediate(
            indices_new_shape,
            indices.dtype,
        )
        build_reshape(
            tosa_graph, indices.name, indices_new_shape, indices_reshaped.name
        )

        tosa_graph.addOperator(
            ts_v0_80.TosaOp.Op().GATHER,
            [weights_reshaped.name, indices_reshaped.name],
            [output_name],
            None,
        )

        if len(weights.shape) == 2:
            output_real_shape = [output.shape[0], output.shape[1]]
            build_reshape(tosa_graph, output_name, output_real_shape, output.name)
