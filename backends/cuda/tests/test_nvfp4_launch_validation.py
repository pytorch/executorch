# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch
import triton
import triton.language as tl

from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm as kernels


@pytest.mark.parametrize("k", [1, 15, 17, 31, 33, 49])
def test_activation_rejects_partial_k16_group(k):
    x = torch.empty((2, k), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K must be divisible by 16"):
        kernels.nvfp4_quantize_activation(x)


def test_split_workspace_boundary():
    with mock.patch.object(kernels, "_split_workspace_limit", return_value=64):
        kernels._check_split_workspace(2, 2, 4)
        with pytest.raises(InvalidLaunchParam, match="96 workspace bytes"):
            kernels._check_split_workspace(2, 3, 4)


def test_unsplit_launch_needs_no_partial_workspace():
    with mock.patch.object(
        kernels, "_split_workspace_limit", side_effect=AssertionError("queried")
    ):
        assert kernels._check_split_workspace(1, 4096, 39936) is None


def test_gemm_still_rejects_illegal_short_k_split():
    with pytest.raises(InvalidLaunchParam, match="Illegal NVFP4 split-K"):
        kernels._check_split(80, 4)


@triton.jit
def _offset_probe(
    OUT, M, SPLIT, N: tl.constexpr, BASE: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    index = lane.to(tl.int64) + BASE
    offsets, plane = kernels._partial_offsets(SPLIT, index, M, N)
    tl.store(OUT + lane, offsets)
    tl.store(OUT + BLOCK + lane, index < plane)


@triton.jit
def _constexpr_offset_probe(
    OUT,
    M: tl.constexpr,
    SPLIT: tl.constexpr,
    N: tl.constexpr,
    BASE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _offset_probe(OUT, M, SPLIT, N, BASE, BLOCK)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize(
    "m,n,split,base",
    [
        (1, 128, 1, 124),
        (4096, 39936, 15, 4096 * 39936 - 4),
        (3, 715827886, 2, 2**31 + 6),
    ],
)
def test_partial_offsets_and_masks_are_int64(m, n, split, base, constant):
    out = torch.empty(16, device="cuda", dtype=torch.int64)
    probe = _constexpr_offset_probe if constant else _offset_probe
    probe[(1,)](out, m, split, N=n, BASE=base, BLOCK=8)
    index = torch.arange(base, base + 8, device="cuda", dtype=torch.int64)
    torch.testing.assert_close(out[:8], split * m * n + index, rtol=0, atol=0)
    torch.testing.assert_close(out[8:], (index < m * n).to(torch.int64), rtol=0, atol=0)
