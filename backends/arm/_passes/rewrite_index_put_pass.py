# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Iterable, List, Sequence, Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.fuse_view_copy_transform_pass import (
    FuseViewCopyTransformPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue


def get_index_put_ops(op):
    if op == exir_ops.edge.aten.index_put.default:
        return (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.backend.tosa.SCATTER.default,
            exir_ops.edge.aten.full.default,
        )
    raise RuntimeError(f"Can't get index_put decomposition for op {op}")


def calculate_tosa_values(
    index_shape: list[int],
    index_nodes: list[Any],
    source_shape: list[int],
) -> tuple[int, int, int, int]:
    # Calculate K, W, C
    # N - kept to 1 for generic n-dim implementation
    # W - the number of positions being updated
    N, K, W, C = 1, 1, 1, 1

    W = math.prod(index_shape)

    for i, dim in enumerate(source_shape):
        if i < len(index_nodes):
            K *= dim

    total_vals = math.prod(source_shape)
    C = int(total_vals / K)

    return N, K, W, C


def calculate_values_stride(source_shape: list[int]) -> list[int]:
    """Calculate strides for a flattened view of the source tensor that are
    multiplied with the indices to build the [N, W] tensor."""
    values_strides: list[int] = []
    stride = 1
    for dim in reversed(source_shape):
        values_strides.insert(0, stride)
        stride *= dim

    return values_strides


class RewriteIndexPutPass(ArmPass):
    """
    This pass transforms index_put operations into TOSA-compatible scatter operations by:
    1. Expanding None indices into explicit range tensors
    2. Calculating flattened index positions
    3. Reshaping tensors into a 3D layout [N, K, C] required by TOSA SCATTER
    4. Applying the scatter operation and reshaping back to the original shape

    Example:
        For source[i, :, j] = values, this pass:
        - Expands ':' to arange(0, dim_size)
        - Calculates flat indices: i * stride[0] + expanded_range * stride[1] + j * stride[2]
        - Reshapes to 3D, applies scatter, reshapes back
    """

    _passes_required_after: Set[Type[ExportPass]] = {FuseViewCopyTransformPass}

    def _expand_none_indices(
        self,
        source_shape: Sequence[int],
        indices: Iterable[Any],
        meta: NodeMetadata,
        full_op,
    ) -> List[ProxyValue]:
        """Replace None indices with explicit ranges."""
        expanded: List[ProxyValue] = []
        for dim_idx, idx in enumerate(indices):
            if idx is None:
                end_index = int(source_shape[dim_idx])
                # Use arange via call to edge operator since full can't create ranges
                full_range = super().call_operator(
                    exir_ops.edge.aten.arange.start_step,
                    (0, end_index, 1),
                    {},
                    meta,
                    updated=True,
                )
                expanded.append(full_range)
            elif not isinstance(idx, ProxyValue):
                raise NotImplementedError(
                    "index_put indices must be tensor ProxyValues or None"
                )
            else:
                expanded.append(idx)
        return expanded

    def _calculate_flat_indices(
        self,
        indices: Sequence[ProxyValue],
        source_shape: Sequence[int],
        num_channels: int,
        ops: Sequence[Any],
        full_op: Any,
        meta: NodeMetadata,
    ) -> ProxyValue:
        """
        The flat index is computed as: sum(index[i] * (stride[i] / num_channels)) for each dimension i.
        """

        values_strides = calculate_values_stride(list(source_shape))
        mul_op, add_op = ops

        new_indices: ProxyValue | None = None
        for i, index_val in enumerate(indices):
            # Get the shape of this dimension's indices to create matching constant tensor
            scale_val = int(values_strides[i] / num_channels)

            # Create constant tensor directly: full([numel], scale_val)
            mul_const = super().call_operator(
                full_op,
                ((1,), scale_val),
                {},
                meta,
                True,
            )

            # Multiply indices by their stride constant
            mul_node = super().call_operator(
                mul_op, (index_val, mul_const), {}, meta, True
            )

            # Accumulate contributions from each dimension
            if new_indices is None:
                new_indices = mul_node
            else:
                new_indices = super().call_operator(
                    add_op, (mul_node, new_indices), {}, meta, True
                )

        if new_indices is None:
            raise RuntimeError("No indices were provided for index_put")

        return new_indices

    def call_operator(self, op, args, kwargs, meta):
        if op not in (exir_ops.edge.aten.index_put.default,):
            return super().call_operator(op, args, kwargs, meta)

        (reshape_op, add_op, mul_op, scatter_op, full_op) = get_index_put_ops(op)

        source, indices, values = args[:3]
        if not isinstance(indices, (list, tuple)):
            raise NotImplementedError("index_put indices must be provided as a tuple")

        source_tensor = source.data
        source_shape = list(source_tensor.shape)

        plain_meta_dict = dict(meta.data)
        plain_meta_dict["input_qparams"] = {}
        plain_meta_dict["output_qparams"] = {}
        plain_meta = NodeMetadata(plain_meta_dict)

        index_dtype = None
        for idx in indices:
            if idx is not None:
                if not isinstance(idx, ProxyValue):
                    raise NotImplementedError(
                        "index_put indices must be tensor ProxyValues or None"
                    )
                index_dtype = idx.data.dtype
                break
        if index_dtype is None:
            raise NotImplementedError(
                "index_put with only None indices is not supported"
            )

        processed_indices = self._expand_none_indices(
            source_shape, indices, plain_meta, full_op
        )

        N, K, W, C = calculate_tosa_values(
            list(processed_indices[0].data.shape),
            [idx.node for idx in processed_indices],
            source_shape,
        )

        indices_reshaped = self._calculate_flat_indices(
            processed_indices,
            source_shape,
            C,
            (mul_op, add_op),
            full_op,
            plain_meta,
        )

        # Scatter expects a 3D layout; flatten everything into [N, K, C].
        reshape_indices = super().call_operator(
            reshape_op, (indices_reshaped, [N, W]), {}, plain_meta, True
        )
        reshape_source = super().call_operator(
            reshape_op, (source, [N, K, C]), {}, meta, True
        )
        reshape_values = super().call_operator(
            reshape_op, (values, (N, W, C)), {}, meta, True
        )
        scatter_node = super().call_operator(
            scatter_op,
            (reshape_source, reshape_indices, reshape_values),
            {},
            meta,
            True,
        )
        return super().call_operator(
            reshape_op, (scatter_node, source_shape), kwargs, meta, True
        )
