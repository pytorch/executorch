# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Sequence, Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.convert_expand_copy_to_repeat import (
    ConvertExpandCopyToRepeatPass,
)
from executorch.backends.arm._passes.fuse_view_copy_transform_pass import (
    FuseViewCopyTransformPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue


def calculate_data_stride(destination_shape: list[int]) -> list[int]:
    """Calculate strides for a flattened view of the destination tensor that are
    multiplied with the indices to build the [N, W] tensor.
    """
    data_strides: list[int] = []
    stride = 1
    for dim in reversed(destination_shape):
        data_strides.insert(0, stride)
        stride *= dim

    return data_strides


class RewriteIndexPutPass(ArmPass):
    """
    This pass transforms index_put with arguments
        - destination, of shape (*K_i, *C_j)
            where *K_i means some number of dims >1 explicitly indexed (copy some entries from data to destination).
            *C_j means some number of dims >=0 fully indexed (copy entire dim from data to destination)
        - indices_tensor_list, a list containing len(*K_i) tensors with shape (W or 1,), indices for each dim K_i.
            W is the number of explicit indexes.
            Indicies_tensors are required to not be None.
        - data, of shape (*C_d_j)
            where len(*C_d_j) can be <= len(C_j)+1 and
                *C_d_j can be broadcast into (W, *C_j),
        - accumulate = False

    The lowering strategy is as follows:
        - Flatten *K_i into K = prod(*K_i)
        - Flatten *C_j int C = prod(*C_j)
        - source_flattened = reshape(source, [N=1, K, C])
        - index_flattened = _calculate_flat_indices()
        - data_broadcast = expand(data, [W, *C_j])
        - data_flattened = reshape(data_broadcast, [N=1, W, C])
        - Apply TOSA.SCATTER(source_flattened, index_flattened, data_flattened)
        - Reshape output back to original destination shape
    """

    def __init__(self):
        super().__init__()
        self.reshape_op = exir_ops.edge.aten.view_copy.default
        self.expand_op = exir_ops.edge.aten.expand_copy.default
        self.add_op = exir_ops.edge.aten.add.Tensor
        self.mul_op = exir_ops.edge.aten.mul.Tensor
        self.scatter_op = exir_ops.backend.tosa.SCATTER.default
        self.full_op = exir_ops.edge.aten.full.default

    _passes_required_after: Set[Type[ExportPass]] = {
        FuseViewCopyTransformPass,
        ConvertExpandCopyToRepeatPass,
    }

    def _calculate_flat_indices(
        self,
        indices: Sequence[ProxyValue],
        shape: list[int],
        meta: NodeMetadata,
    ) -> ProxyValue:
        """
        The flat index is computed as:
        sum(index[i] * (stride[i])) for each dimension i in shape.
        """

        data_strides = calculate_data_stride(shape)

        new_indices: ProxyValue | None = None
        W = 1
        for i, index_val in enumerate(indices):
            # Get the shape of this dimension's indices to create matching constant tensor
            scale_val = int(data_strides[i])

            # Create constant tensor directly: full([numel], scale_val)
            mul_const = super().call_operator(
                self.full_op,
                ((1,), scale_val),
                {
                    "dtype": index_val.data.dtype,
                    "device": index_val.data.device,
                },
                meta,
                True,
            )

            # Multiply indices by their stride constant
            mul_node = super().call_operator(
                self.mul_op, (index_val, mul_const), {}, meta, True
            )

            # Accumulate contributions from each dimension
            if new_indices is None:
                new_indices = mul_node
            else:
                new_indices = super().call_operator(
                    self.add_op, (mul_node, new_indices), {}, meta, True
                )
            assert new_indices is not None
            W = max(new_indices.data.shape[0], 1)

        return super().call_operator(
            self.reshape_op, (new_indices, (1, W)), {}, meta, True
        )

    def call_operator(self, op, args, kwargs, meta, updated: bool | None = False):
        if op not in (exir_ops.edge.aten.index_put.default,):
            return super().call_operator(op, args, kwargs, meta)

        destination, indices_tensor_list, data = args[:3]
        accumulate = len(args) > 3 and bool(args[3])
        if accumulate:
            raise RuntimeError(
                "Encountered index_put with accumulate=True, this is assumed to be handled by an earlier pass."
            )

        indices_tensor_list = list(indices_tensor_list)
        num_explicit_indices = len(indices_tensor_list)
        if any(index_tensor is None for index_tensor in indices_tensor_list):
            raise RuntimeError(
                "Encountered None indices in RewriteIndexPutPass. "
                "Run NormalizeIndexPutNoneIndicesPass before this pass."
            )

        destination_shape = destination.data.shape

        K_i = destination_shape[:num_explicit_indices]
        C_j = destination_shape[num_explicit_indices:]
        C_j = torch.broadcast_shapes(C_j)
        K = math.prod(K_i)
        C = math.prod(C_j)
        # A shape is a tensor of rank 1 -> rank of shape is shape[0].
        indices_shapes = [tuple(idx.data.shape) for idx in indices_tensor_list]
        W = torch.broadcast_shapes(*indices_shapes)[0]

        # CALCULATE FLATTENED DESTINATION [N=1, K, C]
        destination_flattened = super().call_operator(
            self.reshape_op, (destination, [1, K, C]), {}, meta, True
        )

        # CALCULATE FLATTENED INDEX [N=1, W]
        plain_meta_dict = dict(meta.data)
        plain_meta_dict["input_qparams"] = {}
        plain_meta_dict["output_qparams"] = {}
        plain_meta = NodeMetadata(plain_meta_dict)
        indices_flattened = self._calculate_flat_indices(
            indices_tensor_list, K_i, plain_meta
        )

        # CALCULATE FLATTENED DATA [N=1, W, C]
        data_broadcast = super().call_operator(
            self.expand_op,
            (
                data,
                (
                    W,
                    *C_j,
                ),
            ),
            {},
            meta,
            updated=True,
        )
        data_flattened = super().call_operator(
            self.reshape_op, (data_broadcast, (1, W, C)), {}, meta, updated=True
        )

        # DO SCATTER
        scatter_node = super().call_operator(
            self.scatter_op,
            (destination_flattened, indices_flattened, data_flattened),
            {},
            meta,
            True,
        )

        # RESHAPE BACK TO ORIGINAL SHAPE
        out = super().call_operator(
            self.reshape_op, (scatter_node, destination_shape), kwargs, meta, True
        )

        return out
