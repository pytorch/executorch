# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Export-time NVFP4 prefill output policy; decode keeps its input dtype."""

import os

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

NVFP4_DECODE_MAX_ROWS = 4
# Under ET_CUDA_NVFP4_PREFILL_IMPL=auto (see nvfp4_dispatch._dequant_prefill),
# without native FP4 and with input-dtype output, a prefill M statically <= this
# (inclusive) keeps the fused W4A16 kernel; any other M (including a dynamic M
# with a larger upper bound) takes whole-weight dequant + matmul.
NVFP4_FUSED_PREFILL_MAX_ROWS = 64


@torch._dynamo.assume_constant_result
def nvfp4_prefill_fp32_enabled():
    value = os.environ.get("ET_CUDA_NVFP4_PREFILL_OUTPUT", "input")
    if value not in ("input", "fp32"):
        raise ValueError("ET_CUDA_NVFP4_PREFILL_OUTPUT must be input or fp32")
    return value == "fp32"


def nvfp4_prefill_output_dtype(x):
    if nvfp4_prefill_fp32_enabled() and not statically_known_true(
        x.shape[0] <= NVFP4_DECODE_MAX_ROWS
    ):
        return torch.float32
    return None
