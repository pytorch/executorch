# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 linears for small static M (2..16): one op per M bucket.

Packed decode steps (several sessions, or a speculative verify block) run the
same weights for a handful of rows. Reading the weights once per row, as the
M=1 GEMV does, or tiling M like prefill, both waste bandwidth here. Each bucket
op (``triton::nvfp4_small_m_gemm_m{M}``) times three kernels and split-K at AOTI
compile time and keeps the fastest:

- ``rows``: CUDA-core GEMV that decodes each weight tile once and applies it to
  all M rows (activations stay BF16).
- ``w4a4``: native FP4 tensor-core GEMM (``dot_scaled``) on a 16-row tile; the
  activation is quantized to NVFP4 per row (K16 E4M3 + row global scale).
- ``w4a16``: BF16 tensor-core GEMM on a 16-row tile with the weights dequantized
  in registers.

Weights are the same packed NVFP4 tensor for all three. Linears whose
weights must keep BF16 activations (``act_dtype`` other than ``fp4``) use
``triton::nvfp4_small_m_gemm_bf16act_m{M}``, which times only ``rows`` and
``w4a16``.
"""

import os

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_param,
    InvalidLaunchParam,
)
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _check_split,
    _check_static_scale,
    _e4m3_f16,
    _e4m3_to_f32,
    _fp4_pair_f16,
    _has_native_fp4_conversion,
    _quantize_activation,
    _reduce_partials,
    _scaled_pair,
    _SMALL_M_CONFIGS,
    _SOFT_FP4_RESCALE,
    _w4a16,
    _w4a4,
    SPLIT_K_CANDIDATES,
)
from executorch.backends.cuda.triton.kernels.nvfp4_tuning import (
    autotune_do_bench,
    launch_nvfp4_kernel,
    nvfp4_launch_bench,
    prune_nvfp4_stages,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _compile_every_candidate_at_once,
    _QUEUING_ONLY,
    QuantizedGemmFamily,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import wrap_triton

SMALL_M_BUCKETS = (2, 4, 8, 16)
_MODE_STRIDE = 64
_MODE_IDS = {"rows": 0, "w4a4": 1, "w4a16": 2}
# Register budget of the rows kernel's [MB, BN, BK/2] product.
_ROWS_MAX_ELEMENTS = 32768


def _prune_rows(configs, named_args, **kwargs):
    values = {**named_args, **kwargs}
    mb = int(values["MB"])
    fit = [
        c
        for c in configs
        if mb * c.kwargs["BN"] * (c.kwargs["BK"] // 2) <= _ROWS_MAX_ELEMENTS
    ]
    return prune_nvfp4_stages(fit or configs[:1], named_args, **kwargs)


@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn, bk, w in [
            (4, 512, 4),
            (8, 256, 4),
            (8, 512, 4),
            (16, 256, 4),
            (16, 512, 8),
            (32, 256, 8),
        ]
        for st in (1, 2)
    ],
    key=["M", "N", "K", "SPLIT", "MB", "NATIVE", "ROW_PTS", "UNIT_SCALE"],
    prune_configs_by={"early_config_prune": _prune_rows},
    do_bench=autotune_do_bench,
)
@triton.jit
def _gemv_rows(
    X,
    Q,
    S,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    MB: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NATIVE: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
    UNIT_SCALE: tl.constexpr = False,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    rows = tl.arange(0, MB)
    chunk: tl.constexpr = tl.cdiv(K, BK * SPLIT) * BK
    pair = split * (chunk // 2) + tl.arange(0, BK // 2)
    acc = tl.zeros((MB, BN), tl.float32)
    if UNIT_SCALE:
        pts = 1.0
    elif ROW_PTS:
        pts = tl.load(PTS + n, n < N, 0)[:, None]
    else:
        pts = tl.load(PTS)
    for off in range(tl.cdiv(chunk, BK)):
        kk = pair + off * (BK // 2)
        kmask = kk < K // 2
        mask = (n[:, None] < N) & kmask[None, :]
        q = tl.load(Q + n[:, None] * (K // 2) + kk[None, :], mask, 0)
        b = tl.load(S + n[:, None] * (K // 16) + kk[None, :] // 8, mask, 0)
        if NATIVE:
            lo, hi = _scaled_pair(q, _e4m3_to_f32(b, NATIVE), False)
        else:
            lo, hi = _fp4_pair_f16(q)
            sf = _e4m3_f16(b) * _SOFT_FP4_RESCALE
            lo = lo * sf
            hi = hi * sf
        if UNIT_SCALE:
            w0, w1 = lo, hi
        else:
            w0 = (lo * pts).to(X.dtype.element_ty).to(tl.float32)
            w1 = (hi * pts).to(X.dtype.element_ty).to(tl.float32)
        xmask = (rows[:, None] < M) & kmask[None, :]
        x0 = tl.load(X + rows[:, None] * K + kk[None, :] * 2, xmask, 0).to(tl.float32)
        x1 = tl.load(X + rows[:, None] * K + kk[None, :] * 2 + 1, xmask, 0).to(
            tl.float32
        )
        acc += tl.sum(
            w0[None, :, :] * x0[:, None, :] + w1[None, :, :] * x1[:, None, :], 2
        )
    tl.store(
        Y + split * M * N + rows[:, None] * N + n[None, :],
        acc,
        (rows[:, None] < M) & (n[None, :] < N),
    )


_W4A16_SMALL = triton.autotune(
    configs=[
        triton.Config(
            {"BM": 16, "BN": bn, "BK": bk, "GROUP_M": 1}, num_warps=w, num_stages=st
        )
        for bn, bk, w in [
            (32, 128, 4),
            (64, 64, 4),
            (64, 128, 4),
            (64, 256, 8),
            (128, 64, 4),
            (128, 128, 8),
        ]
        for st in (2, 3)
    ],
    key=["M", "N", "K", "SPLIT_K", "NATIVE", "ROW_PTS"],
    prune_configs_by={"early_config_prune": prune_nvfp4_stages},
    do_bench=autotune_do_bench,
)(_w4a16.fn)

_W4A4_SMALL = triton.autotune(
    configs=[c for c in _SMALL_M_CONFIGS if c.kwargs["BM"] == 16],
    key=["M", "N", "K", "SPLIT_K", "ROW_PTS"],
    prune_configs_by={"early_config_prune": prune_nvfp4_stages},
    do_bench=autotune_do_bench,
)(_w4a4.fn)


def _workspace(x, m, n, split, dtype):
    out = torch.empty((m, n), device=x.device, dtype=dtype)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        if split > 1
        else out
    )
    return out, partial


def _reduce(partial, g, pts, out, m, n, split, fp4_act):
    if split > 1 and not _QUEUING_ONLY.get():
        wrap_triton(_reduce_partials)[(triton.cdiv(m * n, 256),)](
            partial,
            g,
            pts,
            out,
            m,
            n,
            split,
            fp4_act,
            256,
            ROW_PTS=pts.numel() != 1,
        )


def launch_rows(bucket, x, qdata, scale, pts, unit_scale, split):
    m, k = x.shape
    n = qdata.shape[0]
    if split not in SPLIT_K_CANDIDATES or split > triton.cdiv(k, 128):
        raise InvalidLaunchParam(f"Illegal NVFP4 rows split-K={split} for K={k}")
    out, partial = _workspace(x, m, n, split, x.dtype)
    launch_nvfp4_kernel(
        _gemv_rows,
        lambda meta: (triton.cdiv(n, meta["BN"]), split),
        (x, qdata, scale, pts, partial, m, n, k, split, bucket),
        5,
        NATIVE=_has_native_fp4_conversion(),
        ROW_PTS=pts.numel() != 1,
        UNIT_SCALE=unit_scale,
    )
    _reduce(partial, pts, pts, out, m, n, split, False)
    return out


def launch_w4a16(x, qdata, scale, pts, split):
    m, k = x.shape
    n = qdata.shape[0]
    _check_split(k, split, m=m, n=n)
    out, partial = _workspace(x, m, n, split, x.dtype)
    launch_nvfp4_kernel(
        _W4A16_SMALL,
        lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]), split),
        (x, qdata, scale, pts, partial, m, n, k, split),
        5,
        NATIVE=_has_native_fp4_conversion(),
        ROW_PTS=pts.numel() != 1,
    )
    _reduce(partial, pts, pts, out, m, n, split, False)
    return out


def launch_w4a4(x, qdata, scale, pts, act_scale, split):
    if not _has_native_fp4_conversion():
        raise InvalidLaunchParam("NVFP4 W4A4 needs native FP4 (SM120+)")
    m, k = x.shape
    n = qdata.shape[0]
    _check_split(k, split, m=m, n=n)
    a, sa, g = _quantize_activation(x, act_scale)
    out, partial = _workspace(x, m, n, split, x.dtype)
    launch_nvfp4_kernel(
        _W4A4_SMALL,
        lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]), split),
        (a, sa, g, qdata, scale, pts, partial, m, n, k, split),
        7,
        ROW_PTS=pts.numel() != 1,
    )
    _reduce(partial, g, pts, out, m, n, split, True)
    return out


def _enabled_modes():
    names = os.environ.get("ET_CUDA_NVFP4_SMALL_M_MODES", "rows,w4a4,w4a16").split(",")
    unknown = [n for n in names if n not in _MODE_IDS]
    if unknown or not names:
        raise ValueError(f"ET_CUDA_NVFP4_SMALL_M_MODES: unknown modes {unknown}")
    # "rows" first: the untimed default must be legal for every input.
    return sorted({_MODE_IDS[n] for n in names})


# Split-K 32 never wins at these shapes; dropping it keeps AOTI compile time sane.
SMALL_M_SPLITS = tuple(s for s in SPLIT_K_CANDIDATES if s <= 16)
_CANDIDATES = tuple(
    mode * _MODE_STRIDE + split for mode in _enabled_modes() for split in SMALL_M_SPLITS
)


def _launch_impl(bucket, x, qdata, scale, pts, act_scale, unit_scale, *, mode_split):
    mode, split = divmod(mode_split, _MODE_STRIDE)
    if mode == _MODE_IDS["rows"]:
        return launch_rows(bucket, x, qdata, scale, pts, unit_scale, split)
    if mode == _MODE_IDS["w4a4"]:
        return launch_w4a4(x, qdata, scale, pts, act_scale, split)
    return launch_w4a16(x, qdata, scale, pts, split)


_launch = autotune_launch_param(
    "mode_split",
    _CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)(_launch_impl)


def _prototype(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    act_scale: torch.Tensor | None,
    unit_scale: bool,
) -> torch.Tensor: ...


def _family_launch(bucket, x, qdata, scale, pts, act_scale, unit_scale):
    # Checked once here: timing runs the W4A4 candidates on random stand-ins.
    _check_static_scale(act_scale, x)
    return _launch(bucket, x.contiguous(), qdata, scale, pts, act_scale, unit_scale)


_launch_bf16_act = autotune_launch_param(
    "mode_split",
    tuple(
        _MODE_IDS[mode] * _MODE_STRIDE + split
        for mode in ("rows", "w4a16")
        for split in SMALL_M_SPLITS
    ),
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)(_launch_impl)


def _family_launch_bf16_act(bucket, x, qdata, scale, pts, act_scale, unit_scale):
    return _launch_bf16_act(
        bucket, x.contiguous(), qdata, scale, pts, act_scale, unit_scale
    )


def _fake(bucket, x, qdata, scale, pts, act_scale, unit_scale):
    return x.new_empty((x.shape[0], qdata.shape[0]))


def _unsupported_reason(bucket, x, qdata, scale, pts, act_scale, unit_scale):
    if x.dim() != 2 or qdata.dim() != 2:
        return "expects a 2-D activation and packed weight"
    if x.dtype not in (torch.bfloat16, torch.float16):
        return f"activation dtype {x.dtype} is not BF16/FP16"
    if x.shape[1] != qdata.shape[1] * 2 or x.shape[1] % 16:
        return "K must match the weight and be a multiple of 16"
    if not statically_known_true(x.shape[0] <= bucket):
        return f"M is not statically <= {bucket}"
    return None


NVFP4_SMALL_M_GEMM = QuantizedGemmFamily(
    "nvfp4_small_m_gemm",
    SMALL_M_BUCKETS,
    _prototype,
    _family_launch,
    _fake,
    _unsupported_reason,
)

NVFP4_SMALL_M_GEMM_BF16_ACT = QuantizedGemmFamily(
    "nvfp4_small_m_gemm_bf16act",
    SMALL_M_BUCKETS,
    _prototype,
    _family_launch_bf16_act,
    _fake,
    _unsupported_reason,
)


def small_m_bucket(x: torch.Tensor, qdata: torch.Tensor):
    """The bucket for a statically 2..16-row activation, else None (M=1 keeps
    the decode GEMV; wider or dynamic M keeps prefill)."""
    if os.environ.get("ET_CUDA_NVFP4_SMALL_M", "1") != "1":
        return None
    if x.dim() != 2 or statically_known_true(x.shape[0] <= 1):
        return None
    return next(
        (
            b
            for b in SMALL_M_BUCKETS
            if statically_known_true(x.shape[0] <= b)
            and _unsupported_reason(b, x, qdata, None, None, 0.0, False) is None
        ),
        None,
    )
