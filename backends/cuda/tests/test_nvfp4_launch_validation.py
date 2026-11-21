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
    "op", [kernels.nvfp4_w4a4, kernels.nvfp4_w4a16, kernels.nvfp4_gemv]
)
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
def test_rejects_malformed_operands(invalid, op):
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
        op(x, q, s, pts)


@pytest.mark.parametrize(
    "static",
    [torch.ones(2), torch.ones(1, dtype=torch.float64), torch.ones(1, device="meta")],
)
def test_activation_rejects_invalid_static_scale(static):
    with pytest.raises(ValueError, match="static activation scale"):
        kernels.nvfp4_quantize_activation(torch.empty((2, 16)), static)


@pytest.mark.parametrize("invalid", ["float_ids", "device"])
def test_embedding_rejects_malformed_ids(invalid):
    q = torch.empty((5, 32), dtype=torch.uint8)
    s = torch.empty((5, 4), dtype=torch.uint8)
    ids = (
        torch.tensor([1.0])
        if invalid == "float_ids"
        else torch.tensor([1], device="meta")
    )
    with pytest.raises(ValueError, match="NVFP4 embedding ids"):
        kernels.nvfp4_embedding(ids, q, s, torch.ones(1), torch.bfloat16)


def test_embedding_skips_id_check_under_fake_tensors():
    # Compiled graphs trace with fake ids, whose values are unknown.
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    ids = mode.from_tensor(torch.tensor([[0, 99]]))
    q, s, pts = (
        mode.from_tensor(t)
        for t in (
            torch.zeros((5, 32), dtype=torch.uint8),
            torch.zeros((5, 4), dtype=torch.uint8),
            torch.ones(1),
        )
    )
    with mode:
        out = kernels.nvfp4_embedding(ids, q, s, pts, torch.bfloat16)
    assert out.shape == (1, 2, 64) and out.dtype == torch.bfloat16


def symbolic_operands(upper):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    mode = FakeTensorMode(shape_env=ShapeEnv())
    x = dynamic_rows(mode, torch.empty(17, 64, dtype=torch.bfloat16), upper)
    weight = [
        mode.from_tensor(t)
        for t in (
            torch.empty((3, 32), dtype=torch.uint8),
            torch.empty((3, 4), dtype=torch.uint8),
            torch.ones(1),
        )
    ]
    return mode, x, weight


@pytest.mark.parametrize(
    "op,launch", [("gemv", "_launch_gemv"), ("w4a16", "_launch_w4a16")]
)
@pytest.mark.parametrize("upper", [None, 64])
def test_dynamic_m_bound_reaches_the_autotuned_launch(op, launch, upper):
    mode, x, weight = symbolic_operands(upper)
    call = kernels.nvfp4_gemv if op == "gemv" else kernels.nvfp4_w4a16
    with mode, mock.patch.object(
        kernels,
        launch,
        side_effect=lambda x, q, *args, **kwargs: x.new_empty((x.shape[0], q.shape[0])),
    ) as launched:
        if op == "gemv" and upper is None:
            # Grid axis 1 holds M, so the GEMV needs a bounded M.
            with pytest.raises(ValueError, match="65535 rows"):
                call(x, *weight)
            return
        call(x, *weight)
    assert launched.call_args.args[4] == upper


def test_gemv_rejects_more_rows_than_grid_axis_1():
    x = torch.empty((65536, 64), dtype=torch.bfloat16)
    q = torch.empty((3, 32), dtype=torch.uint8)
    s = torch.empty((3, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="65535 rows"):
        kernels.nvfp4_gemv(x, q, s, torch.ones(1))


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


def test_gemv_workspace_rejected_before_output_allocation():
    x = torch.empty((2, 80), dtype=torch.bfloat16)
    q = torch.empty((3, 40), dtype=torch.uint8)
    s = torch.empty((3, 5), dtype=torch.uint8)
    pts = torch.ones(1)
    with (
        mock.patch.object(kernels, "_split_workspace_limit", return_value=1),
        mock.patch.object(torch, "empty", side_effect=AssertionError("allocated")),
        pytest.raises(InvalidLaunchParam, match="768 workspace bytes"),
    ):
        kernels._launch_gemv_impl(x, q, s, pts, 2, split_k=32)


@pytest.mark.parametrize("k", [16, 80, 192])
@pytest.mark.parametrize("split", [4, 32])
def test_gemv_keeps_legal_short_k_splits(k, split):
    x = torch.empty((2, k), dtype=torch.bfloat16)
    q = torch.empty((3, k // 2), dtype=torch.uint8)
    s = torch.empty((3, k // 16), dtype=torch.uint8)
    pts = torch.ones(1)
    token = kernels._QUEUING_ONLY.set(True)
    try:
        with (
            mock.patch.object(kernels, "_split_workspace_limit", return_value=2**20),
            mock.patch.object(
                kernels, "_has_native_fp4_conversion", return_value=False
            ),
            mock.patch.object(kernels, "launch_nvfp4_kernel") as launch,
        ):
            out = kernels._launch_gemv_impl(x, q, s, pts, 2, split_k=split)
        assert out.shape == (2, 3)
        assert launch.call_args.args[2][-1] == split
    finally:
        kernels._QUEUING_ONLY.reset(token)


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
