# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-ignore-all-errors

import torch

from torch.library import impl, Library

fallback_op_lib = Library("llama", "DEF")
# registering an operator.
fallback_op_lib.define("fallback(Tensor input) -> Tensor")


@impl(fallback_op_lib, "fallback")
def fallback_impl(a: torch.Tensor) -> torch.Tensor:
    return a


# registering the out variant.
fallback_op_lib.define("fallback.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)")


@impl(fallback_op_lib, "fallback.out")
def fallback_out_impl(a: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(a)
    return out
