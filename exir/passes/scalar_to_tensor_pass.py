# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import ExportPass, map_args, PassResult, ProxyValue


class ScalarToTensorPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()
        self._modified: bool = False

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        # A float scalar next to a float64 tensor must not be rounded to float32
        # before the op promotes it back to float64.
        has_float64_tensor = any(
            isinstance(a, ProxyValue)
            and a.is_tensor()
            and a.to_tensor().dtype == torch.float64
            for a in (*args, *kwargs.values())
        )

        # pyre-ignore
        def try_coerce(value, arg):
            # Note: we want to create tensor constants instead of
            # FakeTensor or ProxyTensor. If python_dispatcher is enabled,
            # the fake_tensor_mode of inputs will be used so that we won't
            # get a constant tensor with torch.tensor() call but instead
            # a fake tensor is created.
            with torch.utils._python_dispatch._disable_current_modes():
                if isinstance(value, (float, int, bool)) and isinstance(
                    arg.type, torch.TensorType
                ):
                    self._modified = True
                    dtype = (
                        torch.float64
                        if isinstance(value, float) and has_float64_tensor
                        else None
                    )
                    return torch.tensor(value, dtype=dtype)

                return value

        args, kwargs = map_args(op, try_coerce, args, kwargs)
        return super().call_operator(op, args, kwargs, meta)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._modified = False
        graph_module = super().call(graph_module).graph_module
        modified = self._modified
        self._modified = False
        return PassResult(graph_module, modified)
