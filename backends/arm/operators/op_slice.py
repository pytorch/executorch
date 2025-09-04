# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from torch.fx import Node


def _fixup_start(start, shape, dim):
    if start.number < 0:
        return start.number % shape[dim]
    else:
        return start.number


def _fixup_end(end, shape, dim):
    if end.number < 0:
        return end.number % shape[dim]
    else:
        return min(end.number, shape[dim])


@register_node_visitor
class SliceVisitor(NodeVisitor):
    target = "aten.slice_copy.Tensor"

    tosa_specs = NodeVisitor.tosa_specs

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

        validate_num_inputs(self.target, inputs, [4, 5])
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [ts.DType.INT8, ts.DType.INT32, ts.DType.FP32],
            output.tosa_spec,
        )

        # See slice_copy_support.py
        if not (len(inputs) == 4 or (len(inputs) == 5 and inputs[4].number == 1)):
            raise ValueError("Unsupported combination of inputs")

        # aten.slice_copy supports slicing in 1d at a time.
        # The arguments are the actual input, dimension of slicing, start index, end index and optinal step or stride.
        input_node, dim, start, end = inputs

        # Translate and check parameters in Pytorch dim order.
        shape = input_node.shape
        dim = dim.number

        start_index = _fixup_start(start, shape, dim)
        end_index = _fixup_end(end, shape, dim)
        size = end_index - start_index

        if size <= 0:
            raise ValueError(
                f"The calculated slice size must be positive. Got {size=} "
                f"with {start_index=} and {end_index=}."
            )
        if size > shape[dim]:
            raise ValueError(
                f"The calculated slice size cannot be greater than the dimension size"
                f". Got {size=} and {shape[dim]=}."
            )

        # Convert aten args to Tosa's start and size shape_t tensors and in TOSA dim order.
        starts = [
            _fixup_start(start, shape, dim) if i == dim else 0
            for i in input_node.dim_order
        ]

        if len(starts) != 0:
            starts_len = len(starts)
        else:
            starts_len = 1
            starts = [0]

        start_tensor = tosa_graph.addConst(
            (starts_len,),
            ts.DType.SHAPE,
            starts,
            node.name + "_start_shape",
        )

        sizes = [size if i == dim else shape[i] for i in input_node.dim_order]
        if len(sizes) != 0:
            sizes_len = len(starts)
        else:
            sizes_len = 1
            sizes = [0]
        sizes_tensor = tosa_graph.addConst(
            (sizes_len,), ts.DType.SHAPE, sizes, node.name + "_sizes_shape"
        )

        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().SLICE,
            [input_node.name, start_tensor.name, sizes_tensor.name],
            [output.name],
            None,
        )
