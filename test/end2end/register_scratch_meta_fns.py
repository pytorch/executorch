# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Register the functions to return metadata for scratch tensors by attaching the
get_scratch_metas method to the out varaint op.

This is just an demo to show the usage. test/end2end/test_end2end.py imports this module.
Instead of just putting the content of this file inside test/end2end/test_end2end.py,
create a separate file to better illustrate the idea and be more similar to implementation
in real use cases.
"""
from typing import Dict, Optional

import torch
from executorch.exir.operator.manip import (
    attach_get_scratch_metas_fn,
    ScratchTensorMetadata,
)
from executorch.exir.tensor import TensorSpec

torch.ops.load_library("//executorch/kernels/portable:custom_ops_generated_lib")


@attach_get_scratch_metas_fn(torch.ops.aten.linear.out)
def linear_out_get_scratch_metas(
    input: TensorSpec,
    weight: TensorSpec,
    bias: Optional[TensorSpec],
    *,
    out: TensorSpec,
) -> Dict[str, ScratchTensorMetadata]:
    return {
        # The key should be exactly the same as the argument name of the scratch tensor specified
        # on the scratch op schema.
        "_scratch_tensor": ScratchTensorMetadata(
            # the scratch tensor has the same dtype as the input tensor
            dtype=input.dtype,
            shape=torch.Size([input.shape[0], weight.shape[0]]),
        )
    }
