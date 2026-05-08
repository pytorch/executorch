# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.rewrite_index_put_pass import RewriteIndexPutPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class NormalizeIndexPutBoolIndexTensorPass(ArmPass):
    """Normalize  single boolean mask index_put scalar to where.
    In the general case, boolean masks are complex and data dependent. The simple case
    x[mask] = scalar
    Can however be directly translated to a where operation:

    out = index_put(destination, [mask], data, accumulate=False)
    becomes
    mask = reshape(mask, mask_shape_padded)
    data = reshape(data, data_shape_padded)
    out = where(mask, data, destination)

    Where the padded shapes are right-padded with ones to match the rank of destination (if needed).
    `data` must be a scalar, to ensure data_padded can be broadcasted to any destination shape
    depending on the (non-constant) mask.
    """

    _passes_required_after: Set[Type[ExportPass]] = {RewriteIndexPutPass}

    def __init__(self):
        super().__init__()
        self.reshape_op = exir_ops.edge.aten.view_copy.default
        self.where_op = exir_ops.edge.aten.where.self

    def _is_valid_bool_mask(
        self,
        indices_tensor_list,
        data,
        accumulate: bool,
    ) -> bool:

        indices = indices_tensor_list[0]
        if indices is None or indices.data.dtype != torch.bool:
            return False

        # We have a boolean mask, validate that the args are supported.
        if accumulate or len(indices_tensor_list) != 1 or data.data.numel() != 1:
            raise RuntimeError(
                f"Got unsupported args for bool mask index_put: {accumulate=}, num indices={len(indices_tensor_list)}!=1, data shape {data.data.shape} not scalar.\n"
                "This is a bug, the operator should not have been delegated."
            )

        return True

    def call_operator(self, op, args, kwargs, meta, updated: bool | None = False):
        if op not in (exir_ops.edge.aten.index_put.default,):
            return super().call_operator(op, args, kwargs, meta, updated)

        destination, indices_tensor_list, data = args[:3]
        accumulate = len(args) > 3 and bool(args[3])
        indices_tensor_list = list(indices_tensor_list)
        if not self._is_valid_bool_mask(indices_tensor_list, data, accumulate):
            return super().call_operator(op, args, kwargs, meta, updated)

        mask = indices_tensor_list[0]
        destination_shape = tuple(destination.data.shape)
        mask_shape = tuple(mask.data.shape)
        padded_mask_shape = (
            *mask_shape,
            *([1] * (len(destination_shape) - len(mask_shape))),
        )

        if len(mask_shape) < len(destination_shape):
            mask = super().call_operator(
                self.reshape_op,
                (mask, padded_mask_shape),
                {},
                meta,
                True,
            )

        if len(destination_shape) != len(data.data.shape):
            data = super().call_operator(
                self.reshape_op,
                (data, [1] * len(destination_shape)),
                {},
                meta,
                True,
            )

        return super().call_operator(
            self.where_op,
            (mask, data, destination),
            kwargs,
            meta,
            True,
        )
