# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Dict

import torch

from executorch.exir.pass_base import ExportPass

from torch._ops import OpOverload


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: Dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
    torch.ops.aten.t.default: torch.ops.aten.t_copy.default,
    torch.ops.aten.view.default: torch.ops.aten.view_copy.default,
    torch.ops.aten.expand.default: torch.ops.aten.expand_copy.default,
    torch.ops.aten.permute.default: torch.ops.aten.permute_copy.default,
    torch.ops.aten.squeeze.default: torch.ops.aten.squeeze_copy.default,
    torch.ops.aten.unsqueeze.default: torch.ops.aten.unsqueeze_copy.default,
    torch.ops.aten.slice.Tensor: torch.ops.aten.slice_copy.Tensor,
}


class ReplaceBrokenOpsWithFunctionalOpsPass(ExportPass):
    """
    TODO: Our backend expects pure functions. However, some operators
    are not functionalized properly. This pass intends to replace
    non-functionalized operators with their functionalized variant.

    TODO: this can be refactors into a general OpReplacementPass
    """

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
            return super().call_operator(
                _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op], args, kwargs, meta
            )
        return super().call_operator(op, args, kwargs, meta)
