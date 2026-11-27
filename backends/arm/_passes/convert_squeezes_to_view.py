# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.fuse_view_copy_transform_pass import (
    FuseViewCopyTransformPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch import SymInt


class ConvertSqueezesToViewPass(ArmOpTargetedPass):
    """Replaces squeeze/unsqueeze operators with view.

    These are simply special cases of the view op, so removing them gives us
    less cases to handle in the node visitiors.

    """

    _passes_required_after: Set[Type[ExportPass]] = {FuseViewCopyTransformPass}
    target_ops = (
        exir_ops.edge.aten.squeeze_copy.dim,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.unsqueeze_copy.default,
    )

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        shape = meta["val"].size()
        if any(isinstance(dim, SymInt) for dim in shape):
            if op == exir_ops.edge.aten.squeeze_copy.dims:
                input_shape = x.data.shape
                dims = {dim % len(input_shape) for dim in args[1]}
                input_shape_proxy = super().call_size_operator_all(x, meta)
                shape = [
                    dim
                    for idx, dim in enumerate(input_shape_proxy)
                    if idx not in dims
                    or isinstance(input_shape[idx], SymInt)
                    or input_shape[idx] != 1
                ]
            if op == exir_ops.edge.aten.unsqueeze_copy.default:
                dim = args[1] % (len(x.data.shape) + 1)
                shape = super().call_size_operator_all(x, meta)
                shape = shape[:dim] + [1] + shape[dim:]

        view_args = (x, list(shape))
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default, view_args, kwargs, meta
        )
