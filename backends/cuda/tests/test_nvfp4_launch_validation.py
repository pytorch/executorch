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
from executorch.backends.cuda.tests.nvfp4_test_utils import dynamic_rows
from executorch.backends.cuda.triton.kernels import nvfp4_quantized_gemm as kernels


@pytest.mark.parametrize("k", [1, 15, 17, 31, 33, 49])
def test_activation_rejects_partial_k16_group(k):
    x = torch.empty((2, k), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K must be divisible by 16"):
        kernels.nvfp4_quantize_activation(x)


@pytest.mark.parametrize(
    "invalid",
    [
        "k_mismatch",
        "scale_k",
        "scale_rows",
        "strided",
        "pts_len",
        "pts_dtype",
        "pts_strided",
    ],
)
def test_w4a4_rejects_malformed_operands(invalid):
    x = torch.empty((2, 64), dtype=torch.bfloat16)
    q = torch.empty((3, 32), dtype=torch.uint8)
    s = torch.empty((3, 4), dtype=torch.uint8)
    pts = torch.ones(1)
    if invalid == "k_mismatch":
        q = torch.empty((3, 16), dtype=torch.uint8)
        s = torch.empty((3, 2), dtype=torch.uint8)
    elif invalid == "scale_k":
        s = torch.empty((3, 2), dtype=torch.uint8)
    elif invalid == "scale_rows":
        s = torch.empty((2, 4), dtype=torch.uint8)
    elif invalid == "strided":
        x = torch.empty((64, 2), dtype=torch.bfloat16).t()
    elif invalid == "pts_len":
        pts = torch.ones(2)
    elif invalid == "pts_dtype":
        pts = torch.ones(1, dtype=torch.bfloat16)
    else:
        pts = torch.ones(1).expand(3)
    with pytest.raises(ValueError, match="NVFP4"):
        kernels.nvfp4_w4a4(x, q, s, pts)


@pytest.mark.parametrize(
    "static",
    [
        torch.ones(2),
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, device="meta"),
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
        torch.tensor([float("inf")]),
        torch.tensor([float("nan")]),
    ],
)
def test_activation_rejects_invalid_static_scale(static):
    x = torch.empty((2, 64), dtype=torch.bfloat16)
    q = torch.empty((3, 32), dtype=torch.uint8)
    s = torch.empty((3, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="static activation scale"):
        kernels.nvfp4_quantize_activation(x, static)
    with pytest.raises(ValueError, match="static activation scale"):
        kernels.nvfp4_w4a4(x, q, s, torch.ones(1), static)


def test_static_scale_values_are_not_read_without_data():
    # Fake and meta tensors carry no values, so even a zero scale passes.
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    x = mode.from_tensor(torch.empty((2, 64), dtype=torch.bfloat16))
    static = mode.from_tensor(torch.zeros((1, 1)))
    with mode:
        kernels._check_static_scale(static, x)
    meta = torch.empty((2, 64), dtype=torch.bfloat16, device="meta")
    kernels._check_static_scale(torch.zeros((1, 1), device="meta"), meta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_scale_values_are_not_read_during_graph_capture():
    x = torch.empty((2, 64), dtype=torch.bfloat16, device="cuda")
    static = torch.zeros((1, 1), device="cuda")
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        kernels._check_static_scale(static, x)


def test_w4a4_requires_native_fp4():
    x = torch.empty((2, 64), dtype=torch.bfloat16)
    q = torch.empty((3, 32), dtype=torch.uint8)
    s = torch.empty((3, 4), dtype=torch.uint8)
    with mock.patch.object(kernels, "_has_native_fp4_conversion", return_value=False):
        with pytest.raises(RuntimeError, match="SM120"):
            kernels.nvfp4_w4a4(x, q, s, torch.ones(1))


def test_split_workspace_boundary():
    with mock.patch.object(kernels, "_split_workspace_limit", return_value=64):
        kernels._check_split_workspace(2, 2, 4)
        with pytest.raises(InvalidLaunchParam, match="96 workspace bytes"):
            kernels._check_split_workspace(2, 3, 4)


def test_split_workspace_uses_dynamic_m_upper_bound():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    def rows(upper):
        mode = FakeTensorMode(shape_env=ShapeEnv())
        return dynamic_rows(mode, torch.empty(2), upper).shape[0]

    bounded = rows(3)
    with mock.patch.object(kernels, "_split_workspace_limit", return_value=64):
        # The hint (2 rows) fits; the upper bound (3 rows) does not.
        with pytest.raises(InvalidLaunchParam, match="96 workspace bytes"):
            kernels._check_split_workspace(2, bounded, 4)
        with pytest.raises(InvalidLaunchParam, match="bounded M"):
            kernels._check_split_workspace(2, rows(None), 4)
        with pytest.raises(InvalidLaunchParam, match="bounded M"):
            kernels._check_split_workspace(2, None, 4)


@pytest.mark.parametrize("upper", [None, 64])
def test_w4a4_passes_traced_m_bound_to_the_autotuned_launch(upper):
    # Timing sees hinted sizes; the bound must come from the traced M so the
    # timed candidates and the final traced call reject the same splits.
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    mode = FakeTensorMode(shape_env=ShapeEnv())
    x = dynamic_rows(mode, torch.empty(17, 64, dtype=torch.bfloat16), upper)
    q, s, pts = (
        mode.from_tensor(t)
        for t in (
            torch.empty((3, 32), dtype=torch.uint8),
            torch.empty((3, 4), dtype=torch.uint8),
            torch.ones(1),
        )
    )
    with mode, mock.patch.object(
        kernels, "_has_native_fp4_conversion", return_value=True
    ), mock.patch.object(
        kernels,
        "_launch_w4a4",
        side_effect=lambda x, q, *args: x.new_empty((x.shape[0], q.shape[0])),
    ) as launch:
        kernels.nvfp4_w4a4(x, q, s, pts)
    assert launch.call_args.args[5] == upper


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
