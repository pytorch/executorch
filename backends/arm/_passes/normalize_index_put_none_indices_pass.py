# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.rewrite_index_put_pass import RewriteIndexPutPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class NormalizeIndexPutNoneIndicesPass(ArmPass):
    """Normalize index_put with None:s in the indices_tensor list by moving
    None-indexed dims to the channel dimensions (*C_j in RewriteIndexPutPass
    teminology) by permutating the destination and data tensors. A None-index
    corresponds to selecting the entire dim, which is equivalent with being a
    channel dimension.

    Example:
    out = index_put(destination, [None, idx1, None, idx2], data)
    becomes
    destination_permuted = permute(destination, destination_dim_order)
    data_front_padded = reshape(data, front_padded_data_shape)
    data_permuted = permute(data, data_dim_order)
    out_permuted = index_put(destination_permuted, [idx1, idx2], data_permuted)
    out = permute(out_permuted, inverse_destination_dim_order)

    Where the permutations of destination and data are decided by how the indexes move.

    Note that None tensors are handled differently in pytorch depending on how many indices tensors there are,
    causing the data tensor to require different shapes, which will require different data permutation.
    Many: all explicit dims are broadcast to a single dim and put in front of data tensor
        destination shape (5,3,4,3) with indices (None, [1,0], None, [0,2]) -> data shape (2, 5, 4)
        Note that this is the behaviour we want! No permutation of data is neccessary.
    One: The explicit dim is kept in place
        destination shape (5,3,4,3) with indices (None, [1,0], None, None) -> data shape (5, 2, 4, 3)
        dim 1 needs to be moved to the front: dim_order = (1,0,2,3).
        This is the same dim order as for the destination tensor.

    """

    _passes_required_after: Set[Type[ExportPass]] = {RewriteIndexPutPass}

    def __init__(self):
        super().__init__()
        self.permute_op = exir_ops.edge.aten.permute_copy.default
        self.reshape_op = exir_ops.edge.aten.view_copy.default

    def _get_data_dim_order(
        self,
        explicit_dims: list[int],
        destination_dim_order: list[int],
    ) -> list[int]:
        """Return dim_order of data tensor."""

        normalized_non_index_dims = destination_dim_order[len(explicit_dims) :]
        data_dim_order = list(range(len(normalized_non_index_dims)))

        if not explicit_dims:
            raise RuntimeError("Expected at least one non-None index tensor.")
        elif len(explicit_dims) > 1:
            # For multiple explicit index tensors, data is already in the order we want.
            return data_dim_order
        else:
            # For single explicit index tensor, use same dim_order as destination
            return destination_dim_order

    def call_operator(self, op, args, kwargs, meta, updated: bool | None = False):
        if op not in (exir_ops.edge.aten.index_put.default,):
            return super().call_operator(op, args, kwargs, meta)

        destination, indices_tensor_list, data = args[:3]
        indices_tensor_list = list(indices_tensor_list)
        if not any(indices_tensor is None for indices_tensor in indices_tensor_list):
            return super().call_operator(op, args, kwargs, meta)

        destination_shape = destination.data.shape
        explicit_dims = [
            dim_idx
            for dim_idx, index_tensor in enumerate(indices_tensor_list)
            if index_tensor is not None
        ]

        none_dims = [
            dim_idx
            for dim_idx, index_tensor in enumerate(indices_tensor_list)
            if index_tensor is None
        ]
        trailing_dims = list(range(len(indices_tensor_list), len(destination_shape)))

        # Handle None indexing of destination tensor.
        destination_dim_order = explicit_dims + none_dims + trailing_dims
        needs_destination_permute = destination_dim_order != list(
            range(len(destination_shape))
        )
        if needs_destination_permute:
            destination = super().call_operator(
                self.permute_op,
                (destination, destination_dim_order),
                {},
                meta,
                updated=True,
            )

        # Handle None indexing of data tensor.
        data_dim_order = self._get_data_dim_order(
            explicit_dims=explicit_dims,
            destination_dim_order=destination_dim_order,
        )
        needs_data_permute = data_dim_order != list(range(len(data_dim_order)))

        if needs_data_permute:
            data_shape = list(data.data.shape)
            aligned_rank = len(data_dim_order)
            if len(data_shape) < aligned_rank:
                # We add dims to data when we move none dims, front pad data with unit dims to match.
                padded_shape = [1] * (aligned_rank - len(data_shape)) + data_shape
                data = super().call_operator(
                    self.reshape_op, (data, padded_shape), {}, meta, updated=True
                )
            data = super().call_operator(
                self.permute_op, (data, data_dim_order), {}, meta, updated=True
            )

        # Call index_put op.
        explicit_indices_tensors = [
            indices_tensor_list[dim_idx] for dim_idx in explicit_dims
        ]
        normalized_args = (destination, explicit_indices_tensors, data, *args[3:])
        out = super().call_operator(op, normalized_args, kwargs, meta, updated=True)

        if not needs_destination_permute:
            return out

        # If needed, reverse permutation of destination tensor.
        inv_dim_order = [0] * len(destination_dim_order)
        for new_dim, original_dim in enumerate(destination_dim_order):
            inv_dim_order[original_dim] = new_dim

        return super().call_operator(
            self.permute_op, (out, inv_dim_order), {}, meta, updated=True
        )
