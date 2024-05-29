# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Import custom op defined in op_llama_cpp_linear_aot.cpp. Those ops are using PyTorch
# C++ APIs for registration so here we need to import the shared library.
# This is only needed for OSS.

import logging
from pathlib import Path

import torch

from torch.library import impl

try:
    op = torch.ops.llama_cpp._weight_int8pack_mm.default
    assert op is not None
except:
    libs = list(Path(__file__).parent.resolve().glob("libcustom_ops_aot_lib.*"))
    assert len(libs) == 1, f"Expected 1 library but got {len(libs)}"
    logging.info(f"Loading custom ops library: {libs[0]}")
    torch.ops.load_library(libs[0])
    op = torch.ops.llama_cpp._weight_int8pack_mm.default
    assert op is not None

custom_ops_lib = torch.library.Library("llama_cpp", "IMPL")

def _validate_params(A: torch.Tensor, B: torch.Tensor, scales: torch.Tensor):
    assert(A.dtype == torch.bfloat16 or A.dtype == torch.float or A.dtype == torch.float16, "Expect A to be either 32-bit or 16-bit float tensor.")
    assert(A.is_contiguous(), "Expect A to be contiguous")
    assert(B.dtype == torch.int8, "Expect B to be int8 tensor.")
    assert(B.is_contiguous(), "Expect B to be contiguous")
    N, K = B.size(0), A.size(1)
    assert(B.size(1) == K, f"Expect B.size(1) == {K}")
    assert(N % 32 == 0 and K % 32 == 0, f"Expect N and K to be multiple of 32 but got {N}, {K}")
    assert(scales.dim() == 1 and scales.size(0) == N, f"Expect scales to be 1d tensor with size {N}")
    assert(scales.dtype == A.dtype, f"Expect scales dtype to be the same as A but got {scales.dtype}")


@impl(custom_ops_lib, "_weight_int8pack_mm", "Meta")
def _llama_cpp_mm_int8_aten_meta(A, B, scales):
    _validate_params(A, B, scales)
    M, N = A.size(0), B.size(0)
    return torch.empty(M, N, dtype=A.dtype, device=A.device, layout=A.layout)
