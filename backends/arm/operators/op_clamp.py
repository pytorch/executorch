# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree


from typing import Any, List, Tuple

import numpy as np
import torch
import tosa_serializer as ts

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
from executorch.backends.arm.tosa.utils import tosa_shape
from torch.fx import Node


@register_node_visitor
class ClampVisitor(NodeVisitor):
    target = "aten.clamp.default"

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
    ]

    def __init__(self, *args):
        super().__init__(*args)

    def _get_min_max_arguments(
        self, node: Node, dtype: torch.dtype
    ) -> Tuple[int | float, int | float]:
        def cast_type(value: Any) -> int | float:
            if isinstance(value, int):
                return value
            else:
                # Attempt to cast to float
                return float(value)

        if dtype.is_floating_point:
            dtype_min = torch.finfo(dtype).min
            dtype_max = torch.finfo(dtype).max
        else:
            dtype_min = torch.iinfo(dtype).min
            dtype_max = torch.iinfo(dtype).max

        min_arg = dtype_min
        max_arg = dtype_max

        if node.args[1] is not None:
            min_arg = cast_type(node.args[1])

        if len(node.args) > 2:
            if node.args[2] is not None:
                max_arg = cast_type(node.args[2])

        return min_arg, max_arg

    def _to_bytes(self, value: int | float, dtype: torch.dtype) -> bytes:
        if dtype == torch.float32:
            return np.frombuffer(np.float32(value).tobytes(), dtype=np.uint8).tolist()
        elif dtype == torch.float16:
            return np.frombuffer(np.float16(value).tobytes(), dtype=np.uint8).tolist()
        elif dtype == torch.int8:
            return np.frombuffer(np.int8(value).tobytes(), dtype=np.uint8).tolist()
        elif dtype == torch.int16:
            return np.frombuffer(np.int16(value).tobytes(), dtype=np.uint8).tolist()
        else:
            raise ValueError(f"Unsupported dtype for to_bytes: {dtype}")

    def define_node(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        validate_num_inputs(self.target, inputs, [2, 3])
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [
                ts.DType.INT8,
                ts.DType.INT16,
                ts.DType.INT32,
                ts.DType.FP16,
                ts.DType.FP32,
            ],
            output.tosa_spec,
        )

        node_input_dtype = node.meta["val"].dtype
        # NOTE: Quantization of the min/max arguments is handled by QuantizeClampArgumentsPass
        min_val, max_val = self._get_min_max_arguments(node, node_input_dtype)

        if inputs[0].dtype == ts.DType.INT32:
            self._define_int32_clamp(
                node, tosa_graph, inputs, output, int(min_val), int(max_val)
            )
            return

        attr = ts.TosaSerializerAttribute()
        attr.ClampAttribute(
            self._to_bytes(min_val, node_input_dtype),
            self._to_bytes(max_val, node_input_dtype),
            nan_mode=ts.NanPropagationMode.PROPAGATE,
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CLAMP,
            [inputs[0].name],
            [output.name],
            attr,
        )

    def _define_int32_clamp(
        self,
        node: Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
        min_val: int,
        max_val: int,
    ) -> None:
        """Lower int32 clamp via MIN/MAX because TOSA lacks int32 CLAMP."""

        broadcast_shape = list(tosa_shape(inputs[0].shape, inputs[0].dim_order))
        const_shape = tuple(broadcast_shape)
        numel = int(np.prod(broadcast_shape)) if broadcast_shape else 1

        max_tensor = tosa_graph.addConst(
            const_shape,
            inputs[0].dtype,
            [max_val] * numel,
            name=f"{output.name}_int32_max",
        )
        min_tensor = tosa_graph.addConst(
            const_shape,
            inputs[0].dtype,
            [min_val] * numel,
            name=f"{output.name}_int32_min",
        )

        intermediate_name = f"{output.name}_int32_tmp"
        tosa_graph.currRegion.currBasicBlock.addTensor(
            intermediate_name,
            list(tosa_shape(output.shape, output.dim_order)),
            output.dtype,
        )

        min_attr = ts.TosaSerializerAttribute()
        min_attr.MinimumAttribute(nan_mode=ts.NanPropagationMode.PROPAGATE)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.MINIMUM,
            [inputs[0].name, max_tensor.name],
            [intermediate_name],
            min_attr,
        )

        max_attr = ts.TosaSerializerAttribute()
        max_attr.MaximumAttribute(nan_mode=ts.NanPropagationMode.PROPAGATE)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.MAXIMUM,
            [intermediate_name, min_tensor.name],
            [output.name],
            max_attr,
        )
