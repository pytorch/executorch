# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from typing import Any, List

import executorch.backends.arm.tosa.utils as tutils

import numpy as np
import tosa_serializer as ts

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_same_dtype,
)
from executorch.backends.arm.tosa.mapping import extract_tensor_meta, TosaArg
from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.fx import Node


@register_node_visitor
class CommonIndexTensorVisitor(NodeVisitor):
    target = "aten.index.Tensor"

    def __init__(self, *args):
        super().__init__(*args)

    def _get_tensor_info(self, tensor: Node):
        """
        Consolidates obtaining name, dtype and shape into a common function
        reconciling access based on the type of the input.

        Args:
            fake_tensors (list[FakeTensor]): List of 2 or more FakeTensors
            who's shapes to evaluate

        Returns:
            tuple[bool, list[int]]: First element is whether the shapes are
            broadcastable. Second element is the common shape if compatible.
            If not, empty list.

        """
        if isinstance(tensor, Node):
            dtype, shape, _ = extract_tensor_meta(tensor.meta, self.tosa_spec)
            return tensor.name, dtype, shape
        else:
            return tensor.name, tensor.dtype, tensor.shape

    def _calculate_tosa_vals(
        self,
        index_shape: List[int],
        index_nodes: List[Node],
        values_shape: List[int],
    ):
        # From TOSA spec
        # N - number of batches
        # W - number of indices in each batch
        # K - range of each index (number of elements to index through)
        # C - number of data channels for each index
        N, K, W, C = 1, 1, 1, 1

        # Calculate K, W, C
        # N - kept to 1 for generic n-dim implementation
        #   Note: If/when slice and ellipsis support is added batching
        #   may have to be used to facilitate proper implementation of
        #   the relevant logic.
        # W - common between all indices as they have been broadcast
        #   to a common shape in a pass.
        W = math.prod(index_shape)

        for i, dim in enumerate(values_shape):
            if i < len(index_nodes):
                K *= dim

        total_vals = math.prod(values_shape)
        C = int(total_vals / K)

        return N, K, W, C

    def _calculate_value_strides(self, values_shape: List[int]) -> List[int]:
        values_strides: List[int] = []
        stride = 1
        for dim in reversed(values_shape):
            values_strides.insert(0, stride)
            stride *= dim

        return values_strides


@register_node_visitor
class IndexTensorVisitor(CommonIndexTensorVisitor):
    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
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
        """
        This approach uses the fact that all indexing tensors are incremented
        simultaneously and they essentially act as a map along the corresponding
        dimensions of the values tensor.
        Note: that this does not hold true when slicing or ellipsis ops
        are involved as such they are not currently not supported.

        As such this approach flattens out the values tensor and
        constructs a flattened out index obtained by flattening out the
        index tensors, multiplying them by the relevant stride and accumulating them.

        This approach suffers from the fact that we are taking a number of index tensors of
        type int32 and applying multiplications and additions.

        If the number of total elements in the values tensor exceeds int32 limits
        then this approach falls apart.
        """

        validate_same_dtype(self.target, [inputs[0], output])

        values, indices = inputs
        index_nodes = indices.special

        # Broadcast indices
        broadcasted_tensors = tutils.broadcast_tensors(
            tosa_graph, index_nodes, self.tosa_spec
        )

        # Calculate strides so we can shift indices down the line.
        values_strides = self._calculate_value_strides(values.shape)

        # The indices have already been broadcast to a common shape
        # in so they are all the same.
        _, index_dtype, index_shape = self._get_tensor_info(broadcasted_tensors[0])

        N, K, W, C = self._calculate_tosa_vals(index_shape, index_nodes, values.shape)

        gather_idx_shape = [N, W]

        gather_index_name = ""
        # Flatten out and shift indexes.
        for i, index_node in enumerate(broadcasted_tensors):
            index_name, _, _ = self._get_tensor_info(index_node)
            index_name = index_node.name

            stride_shifted_indices = tosa_graph.addIntermediate(
                index_shape,
                index_dtype,
            )

            # Division by C is necessary when len(indices) < values.rank
            # When there are dimensions left unindexed that changes the
            # channels and thus the stride-shift.
            data = np.full(index_shape, int(values_strides[i] / C))
            mul_const = tosa_graph.addConst(index_shape, index_dtype, data)
            tosa_graph.addConst([1], ts.DType.INT8, 0, name=f"{node.name}_{i}_shift")
            attr = ts.TosaSerializerAttribute()
            attr.MulAttribute()
            self._serialize_operator(
                node,
                tosa_graph,
                ts.Op.MUL,
                [index_name, mul_const.name, f"{node.name}_{i}_shift"],
                [stride_shifted_indices.name],
                attr,
            )

            reshaped_idxs = tosa_graph.addIntermediate(
                gather_idx_shape,
                index_dtype,
            )
            tutils.build_reshape_tosa_1_0(
                tosa_graph,
                stride_shifted_indices.name,
                gather_idx_shape,
                reshaped_idxs.name,
                shape_name_override=f"{node.name}_{i}_shape",
            )

            # Guarantees that the accumulation tensor is properly
            # initialized and does not contain junk data.
            if i == 0:
                gather_index_name = reshaped_idxs.name
            else:
                add_idxs = tosa_graph.addIntermediate(
                    reshaped_idxs.shape,
                    reshaped_idxs.dtype,
                )
                attr = ts.TosaSerializerAttribute()
                attr.AddAttribute()
                self._serialize_operator(
                    node,
                    tosa_graph,
                    ts.Op.ADD,
                    [gather_index_name, reshaped_idxs.name],
                    [add_idxs.name],
                    attr,
                )
                gather_index_name = add_idxs.name

        gather_vals_shape = [N, K, C]
        reshaped_input = tosa_graph.addIntermediate(gather_vals_shape, values.dtype)

        tutils.build_reshape_tosa_1_0(
            tosa_graph,
            values.name,
            gather_vals_shape,
            reshaped_input.name,
            shape_name_override=f"{node.name}_index_shape",
        )

        gather_out_shape = (N, W, C)
        gather_out = tosa_graph.addIntermediate(
            gather_out_shape,
            output.dtype,
        )
        attr = ts.TosaSerializerAttribute()
        attr.GatherAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.GATHER,
            [reshaped_input.name, gather_index_name],
            [gather_out.name],
            attr,
        )

        output_shape = tutils.tosa_shape(output.shape, output.dim_order)

        tutils.build_reshape_tosa_1_0(
            tosa_graph,
            gather_out.name,
            list(output_shape),
            output.name,
            shape_name_override=f"{node.name}_output_shape",
        )
