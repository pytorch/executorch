# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import (
    ops as exir_ops,
)  # To provide the implementation of the operators
from torch.library import impl, Library, register_fake

# New operator library with a custom namespace to allow fusion etc.
lib = Library("cortex_m", "DEF")

###
# add.Tensor
###

lib.define("aten_add_tensor(Tensor self, Tensor other, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)")

@impl(lib, "aten_add_tensor", "CompositeExplicitAutograd")
def aten_add_tensor_impl(input1, input2, dtype, out):
    return exir_ops.edge.cortex_m.aten_add_tensor.default(input1, input2, dtype, dtype)


###
# add.out
###

lib.define(
    "add.out(Tensor input1, Tensor input2, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)"
)

@impl(lib, "add.out", "CompositeExplicitAutograd")
def add_out_impl(
    input1: torch.Tensor,
    input2: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> torch.Tensor:
    """
    The implementation of cmsis-nn add.out.
    """

    return exir_ops.edge.cortex_m.add.default(
        input1, input2, dtype, dtype
    )
