# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Sequence

import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore
import tosa_tools.v0_80.tosa.Op as TosaOp  # type: ignore

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from torch.fx import Node


def _add_node_to_tosa_graph(
    tosa_graph: ts.TosaSerializer,
    inputs: List[TosaArg],
    output: TosaArg,
    supported_dtypes: Sequence,
) -> None:
    if len(inputs) != 3:
        raise ValueError(f"aten.where.self expects 3 arguments, got {len(inputs)}")

    if inputs[0].dtype is not ts.DType.BOOL:
        raise ValueError("Input 0 needs to have dtype BOOL")
    if inputs[1].dtype != inputs[2].dtype:
        raise ValueError(
            "Non-condition tensors must have same data type, got "
            f"{inputs[1].dtype} and {inputs[2].dtype}"
        )
    for input_ in inputs[1:]:
        if input_.dtype not in supported_dtypes:
            raise ValueError(
                f"Input needs to be of torch dtype {supported_dtypes}, got {input_.dtype}"
            )

    tosa_graph.addOperator(
        TosaOp.Op().SELECT,
        [inputs[0].name, inputs[1].name, inputs[2].name],
        [output.name],
        None,
    )


@register_node_visitor
class WhereVisitor_080_BI(NodeVisitor):
    target = "aten.where.self"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:

        bi_supported_dtypes = [
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.BOOL,
        ]
        _add_node_to_tosa_graph(tosa_graph, inputs, output, bi_supported_dtypes)


@register_node_visitor
class WhereVisitor_080_MI(WhereVisitor_080_BI):

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        mi_supported_dtypes = [
            ts.DType.FP16,
            ts.DType.FP32,
            ts.DType.INT8,
            ts.DType.INT16,
            ts.DType.INT32,
            ts.DType.BOOL,
        ]
        _add_node_to_tosa_graph(tosa_graph, inputs, output, mi_supported_dtypes)
