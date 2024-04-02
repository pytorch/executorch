# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import ExportPass, ProxyValue
from torch.utils import _pytree as pytree


class RemoveNoopPass(ExportPass):
    """
    Removes noops that pass through arguments.
    """

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        if op not in (
            torch.ops.aten.to.dtype,
            torch.ops.aten.dropout.default,
            torch.ops.aten.slice_copy.Tensor,
        ):
            return super().call_operator(op, args, kwargs, meta)

        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        orig_tensor = (
            args[0].to_tensor() if isinstance(args[0], ProxyValue) else args[0]
        )

        if orig_tensor is op(*args_data, **kwargs_data):
            return args[0]

        if op == torch.ops.aten.slice_copy.Tensor:
            result = op(*args_data, **kwargs_data)
            if orig_tensor.size() == result.size():
                return args[0]

        return super().call_operator(op, args, kwargs, meta)
