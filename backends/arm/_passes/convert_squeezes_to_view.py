# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

from executorch.backends.arm._passes import ArmPass

from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ConvertSqueezesToViewPass(ArmPass):
    """
    Replaces squeeze/unsqueeze operators with view. These are simply special cases of the view op, so removing them gives us less cases to handle in the node visitiors.
    """

    _passes_required_after: Set[Type[ExportPass]] = {FuseViewCopyTransform}

    def call_operator(self, op, args, kwargs, meta):
        if op not in [
            exir_ops.edge.aten.squeeze_copy.dims,
            exir_ops.edge.aten.unsqueeze_copy.default,
        ]:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        shape = meta["val"].size()
        view_args = (x, list(shape))
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default, view_args, kwargs, meta
        )
