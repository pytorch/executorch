# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, List

import torch

from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    validate_num_inputs,
    validate_same_dtype,
    validate_valid_dtype,
)
from executorch.backends.arm.tosa.mapping import TosaArg


def permutation_vector_to_matrix(permutation_vector: list[int]) -> torch.Tensor:
    """
    Converts a permutation vector of length N to a NxN matrix that describes the same permutation.
    for example:
    (1,0,2)
    ->
    [0 1 0]
    |1 0 0|
    [0 0 1]
    """
    N = len(permutation_vector)
    P = torch.zeros(N, N)
    for row_index, col_index in enumerate(permutation_vector):
        P[row_index][col_index] = 1
    return P


def permutation_matrix_to_vector(permutation_matrix: torch.Tensor) -> list[int]:
    """
    Converts a NxN permutation matrix to a permutation vector of length N that describes the same permutation.
    [0 1 0]
    |1 0 0|
    [0 0 1]
    ->
    (1,0,2)
    """
    N = len(permutation_matrix)
    if N != len(permutation_matrix[0]):
        raise ValueError(
            f"A permutation matrix must be square, got shape {permutation_matrix.shape}"
        )

    p = [0] * N
    for row_index, row in enumerate(permutation_matrix):
        saw_one = False
        for col_index, value in enumerate(row):
            if value == 1:
                if saw_one:
                    raise ValueError(
                        f"A permutation matrix can only have one 1 per row, got {row=}"
                    )
                p[row_index] = col_index
                saw_one = True
            elif value != 0:
                raise ValueError(
                    f"A permutation matrix only contains 1's and 0's, got {value=}"
                )
    return p


def transform_permutation_vector(permutation_vector: list[int], dim_order: list[int]):
    """Transforms a permutation to dim_order."""

    # We need to first transform to dim_order, apply the permutation P,
    # and then transform back to the original dim_order.
    # This transformation, S, is also a permutation, with the dim_order as permutation vector.

    # To do this, represent P and S with permutation matrices.
    # Matrices can handle chained transformations and inversion easily.
    S = permutation_vector_to_matrix(dim_order)
    # The inverse of a permutation matrix is its transpose.
    S_inverse = S.t()
    P = permutation_vector_to_matrix(permutation_vector)

    # The complete transformation is S * P * S_inverse.
    transformation_matrix = S.matmul(P.matmul(S_inverse))

    # Luckily, since it is just a combination of permutations, the result is also a permutation
    # that can again be described by a new permutation vector.
    permutation_vector = permutation_matrix_to_vector(transformation_matrix)
    return permutation_vector


@register_node_visitor
class PermuteVisitor(NodeVisitor):
    target = "aten.permute_copy.default"

    tosa_specs = NodeVisitor.tosa_specs

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        import serializer.tosa_serializer as ts

        validate_num_inputs(self.target, inputs, 2)
        validate_same_dtype(self.target, [inputs[0], output], ts)
        validate_valid_dtype(
            self.target,
            [inputs[0], output],
            [ts.DType.INT8, ts.DType.INT16, ts.DType.INT32, ts.DType.FP32],
            output.tosa_spec,
        )

        # The permutation vector describes a permutation P in default Pytorch dim_order.
        # For rank 4, the default dim_order NCHW.
        # E.g. (2,3,0,1) -> permute (n,c,h,w) to (w,c,n,h)
        permutation_vector = inputs[1].special

        if output.dim_order != tuple(range(len(output.dim_order))):
            # the permutation vector can't be used directly if we are not in NCHW dim_order.
            # Transform to dim_order.
            permutation_vector = transform_permutation_vector(
                permutation_vector, output.dim_order
            )

        attr = ts.TosaSerializerAttribute()
        attr.TransposeAttribute(permutation_vector)
        self._serialize_operator(
            node,
            tosa_graph,
            ts.TosaOp.Op().TRANSPOSE,
            [inputs[0].name],
            [output.name],
            attr,
        )
