# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import ExportPass, map_args


class ScalarToTensorPass(ExportPass):
    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        # pyre-ignore
        def try_coerce(value, arg):
            # Note: we want to create tensor constants instead of
            # FakeTensor or ProxyTensor. If python_dispatcher is enabled,
            # the fake_tensor_mode of inputs will be used so that we won't
            # get a constant tensor with torch.tensor() call but instead
            # a fake tensor is created.
            with torch.utils._python_dispatch._disable_current_modes():
                return (
                    torch.tensor(value)
                    if isinstance(value, (float, int, bool))
                    and isinstance(arg.type, torch.TensorType)
                    else value
                )

        args, kwargs = map_args(op, try_coerce, args, kwargs)
        return super().call_operator(op, args, kwargs, meta)
