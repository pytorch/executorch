# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

# pyre-unsafe

from typing import Any, List, Tuple

import numpy as np
import torch

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
)

from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification
from torch.fx import Node


@register_node_visitor
class ClampVisitor_080_BI(NodeVisitor):
    target = "aten.clamp.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+BI"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def _create_clamp_node(
        self,
        tosa_graph: Any,
        input_name: str,
        output_name: str,
        min_int: int,
        max_int: int,
        min_fp32: float,
        max_fp32: float,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        attr = ts.TosaSerializerAttribute()
        attr.ClampAttribute(
            tosa_graph.builder,
            min_int,
            max_int,
            min_fp32,
            max_fp32,
        )
        tosa_graph.addOperator(ts.TosaOp.Op().CLAMP, [input_name], [output_name], attr)

    def _get_min_max_arguments(
        self, node: Node, dtype_min: int | float, dtype_max: int | float
    ) -> Tuple[int | float, int | float]:

        def cast_type(value: Any) -> int | float:
            if isinstance(value, int):
                return value
            else:
                # Attempt to cast to float
                return float(value)

        min_arg = dtype_min
        max_arg = dtype_max

        if node.args[1] is not None:
            min_arg = cast_type(node.args[1])

        if len(node.args) > 2:
            if node.args[2] is not None:
                max_arg = cast_type(node.args[2])

        return min_arg, max_arg

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import tosa_tools.v0_80.serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, [2, 3])
        validate_same_dtype(self.target, [inputs[0], output], ts)

        min_int8, max_int8 = self._get_min_max_arguments(
            node,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
        )

        # NOTE: Quantization of the min/max arguments is handled by QuantizeOperatorArguments
        self._create_clamp_node(
            tosa_graph,
            inputs[0].name,
            output.name,
            int(min_int8),
            int(max_int8),
            0,
            0,
        )


@register_node_visitor
class ClampVisitor_080_MI(ClampVisitor_080_BI):
    # inheriting 'target' from BI class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-0.80+MI"),
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
        import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, [2, 3])
        validate_same_dtype(self.target, [inputs[0], output], ts)

        if inputs[0].dtype == ts.DType.INT8:
            # Call the inherited define_node for handling integers
            super().define_node(node, tosa_graph, inputs, output)
        else:
            min_fp32, max_fp32 = self._get_min_max_arguments(
                node,
                torch.finfo(torch.float32).min,
                torch.finfo(torch.float32).max,
            )

            self._create_clamp_node(
                tosa_graph,
                inputs[0].name,
                output.name,
                0,
                0,
                min_fp32,
                max_fp32,
            )


@register_node_visitor
class ClampVisitor_INT(NodeVisitor):
    target = "aten.clamp.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def _get_min_max_arguments(
        self, node: Node, dtype_min: int | float, dtype_max: int | float
    ) -> Tuple[int | float, int | float]:

        def cast_type(value: Any) -> int | float:
            if isinstance(value, int):
                return value
            else:
                # Attempt to cast to float
                return float(value)

        min_arg = dtype_min
        max_arg = dtype_max

        if node.args[1] is not None:
            min_arg = cast_type(node.args[1])

        if len(node.args) > 2:
            if node.args[2] is not None:
                max_arg = cast_type(node.args[2])

        return min_arg, max_arg

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts  # type: ignore

        validate_num_inputs(self.target, inputs, [2, 3])
        validate_same_dtype(self.target, [inputs[0], output], ts)

        # NOTE: Quantization of the min/max arguments is handled by QuantizeOperatorArguments
        min_int8, max_int8 = self._get_min_max_arguments(
            node,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
        )

        attr = ts.TosaSerializerAttribute()
        attr.ClampAttribute(
            tosa_graph.builder,
            np.int8(min_int8).tobytes(),
            np.int8(max_int8).tobytes(),
            nan_mode=1,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().CLAMP, [inputs[0].name], [output.name], attr
        )


@register_node_visitor
class ClampVisitor_FP(ClampVisitor_INT):
    # inheriting 'target' from INT class

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
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

        validate_num_inputs(self.target, inputs, [2, 3])
        validate_same_dtype(self.target, [inputs[0], output], ts)

        min_fp32, max_fp32 = self._get_min_max_arguments(
            node,
            torch.finfo(torch.float32).min,
            torch.finfo(torch.float32).max,
        )

        attr = ts.TosaSerializerAttribute()
        attr.ClampAttribute(
            tosa_graph.builder,
            np.float32(min_fp32).tobytes(),
            np.float32(max_fp32).tobytes(),
            nan_mode=1,
        )

        tosa_graph.addOperator(
            ts.TosaOp.Op().CLAMP, [inputs[0].name], [output.name], attr
        )
