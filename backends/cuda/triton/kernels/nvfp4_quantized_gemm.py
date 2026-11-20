# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 K16/E4M3 CUDA kernels, captured into AOTI through triton_op.

Packed weights use low nibble first. Scales remain N-major; dot_scaled's
lowering owns the architecture-specific scale operand layout.
"""

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.autotune.launch_params import (
    _size_hint,
    autotune_launch_param,
    InvalidLaunchParam,
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
)
from torch.library import triton_op, wrap_triton

SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 32)


@torch._dynamo.assume_constant_result
def _split_workspace_limit():
    return (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory // 2
    )


def _check_split_workspace(split_k, m, n):
    if split_k > 1:
        workspace = split_k * _size_hint(m) * _size_hint(n) * 4
        if workspace > _split_workspace_limit():
            raise InvalidLaunchParam(
                f"NVFP4 split-K={split_k} needs {workspace} workspace bytes; "
                "the autotune limit is half the device memory"
            )


def _check_split(k, split_k, *, m=None, n=None):
    if split_k not in SPLIT_K_CANDIDATES or split_k > triton.cdiv(k, 128):
        raise InvalidLaunchParam(f"Illegal NVFP4 split-K={split_k} for K={k}")
    if m is not None and n is not None:
        _check_split_workspace(split_k, m, n)


@triton.jit
def _fp4(code):
    mag = code & 7
    val = tl.where(
        mag < 2,
        mag.to(tl.float32) * 0.5,
        (((mag >> 1).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
        * (1.0 + (mag & 1).to(tl.float32) * 0.5),
    )
    return tl.where((code & 8) != 0, -val, val)


@triton.jit
def _row_scale(X, G, M, K: tl.constexpr, STATIC: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    g = STATIC if STATIC > 0 else tl.max(tl.abs(x), 0) / (448.0 * 6.0)
    tl.store(G + row, tl.where(g > 0, g, 1.0))


@triton.jit
def _activation(X, G, Q, S, M, K: tl.constexpr, BG: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1) * BG + tl.arange(0, BG)
    k = group[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    g = tl.load(G + row)
    scale = (tl.max(tl.abs(x), 1) / (6.0 * g)).to(tl.float8e4nv)
    sf = scale.to(tl.float32)
    v = x / tl.where(sf[:, None] > 0, sf[:, None] * g, 1.0)
    a = tl.abs(v)
    # Round E2M1 to nearest, ties to even; saturation at 6.
    c = tl.where(a > 0.25, 1, 0)
    c = tl.where(a >= 0.75, 2, c)
    c = tl.where(a > 1.25, 3, c)
    c = tl.where(a >= 1.75, 4, c)
    c = tl.where(a > 2.5, 5, c)
    c = tl.where(a >= 3.5, 6, c)
    c = tl.where(a > 5.0, 7, c)
    c = (c | tl.where(v < 0, 8, 0)).to(tl.uint8)
    pairs = tl.reshape(c, (BG, 8, 2))
    lo, hi = tl.split(pairs)
    q = lo | (hi << 4)
    tl.store(
        Q + row * (K // 2) + group[:, None] * 8 + tl.arange(0, 8)[None, :],
        q,
        group[:, None] < K // 16,
    )
    tl.store(
        S + row * (K // 16) + group, scale.to(tl.uint8, bitcast=True), group < K // 16
    )


@torch._dynamo.assume_constant_result
def _has_native_fp4_conversion():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (4, 8, 16)],
    key=["M", "K", "STATIC"],
    do_bench=autotune_do_bench,
)
@triton.jit
def _fused_activation(
    X, Q, S, G, M, K: tl.constexpr, STATIC: tl.constexpr, BK: tl.constexpr
):
    # One row load supplies both global amax and K16 scales. The native
    # conversion packs two E2M1 values in one instruction, low nibble first.
    row = tl.program_id(0)
    group = tl.arange(0, BK // 16)
    idx = group[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(X + row * K + idx, idx < K, 0).to(tl.float32)
    block_max = tl.max(tl.abs(x), 1)
    g = STATIC if STATIC > 0 else tl.max(block_max, 0) / (448.0 * 6.0)
    g = tl.where(g > 0, g, 1.0)
    sf = (block_max / (6.0 * g)).to(tl.float8e4nv)
    v = x / tl.where(
        sf.to(tl.float32)[:, None] > 0, sf.to(tl.float32)[:, None] * g, 1.0
    )
    # Match the software threshold quantizer for signed zero and NaN.
    v = tl.where((v == v) & (v != 0), v, 0.0)
    lo, hi = tl.split(tl.reshape(v, (BK // 16, 8, 2)))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; mov.b32 $0, {tmp,0,0,0}; }",
        constraints="=r,f,f",
        args=[lo, hi],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    tl.store(
        Q + row * (K // 2) + group[:, None] * 8 + tl.arange(0, 8)[None, :],
        packed,
        group[:, None] < K // 16,
    )
    tl.store(
        S + row * (K // 16) + group, sf.to(tl.uint8, bitcast=True), group < K // 16
    )
    tl.store(G + row, g)


@triton_op("triton::nvfp4_quantize_activation", mutates_args={})
def nvfp4_quantize_activation(
    x: torch.Tensor, static_scale: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = x.shape
    if k % 16:
        raise ValueError("NVFP4 activation K must be divisible by 16")
    q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty((m, k // 16), device=x.device, dtype=torch.uint8)
    g = torch.empty((m,), device=x.device, dtype=torch.float32)
    if _has_native_fp4_conversion():
        launch_nvfp4_kernel(
            _fused_activation,
            (m,),
            (x, q, s, g, m, k, static_scale, triton.next_power_of_2(k)),
            4,
        )
    else:
        wrap_triton(_row_scale)[(m,)](
            x, g, m, k, static_scale, triton.next_power_of_2(k)
        )
        wrap_triton(_activation)[(m, triton.cdiv(k // 16, 32))](
            x, g, q, s, m, k, 32, num_warps=4
        )
    return q, s, g


_SMALL_M_CONFIGS = [
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=3
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 256, "GROUP_M": 1}, num_warps=2, num_stages=3
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 256, "GROUP_M": 1}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 256, "GROUP_M": 1}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"BM": 32, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 128, "BK": 256, "GROUP_M": 1}, num_warps=2, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 128, "BK": 256, "GROUP_M": 1}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=3
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=2
    ),
]
_SMALL_M_CONFIGS += [
    triton.Config(
        {"BM": 16, "BN": 128, "BK": 256, "GROUP_M": 1}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 128, "BK": 256, "GROUP_M": 1}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 256, "GROUP_M": 1}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 256, "GROUP_M": 1}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BM": 16, "BN": 64, "BK": 512, "GROUP_M": 1}, num_warps=8, num_stages=1
    ),
    triton.Config(
        {"BM": 32, "BN": 32, "BK": 512, "GROUP_M": 1}, num_warps=2, num_stages=1
    ),
]


def _prune_w4a4(configs, named_args, **kwargs):
    values = {**named_args, **kwargs}
    eligible = (
        configs
        if _size_hint(values["M"]) <= 4
        else [c for c in configs if c not in _SMALL_M_CONFIGS]
    )
    return prune_nvfp4_stages(eligible, named_args, **kwargs)


@triton.autotune(
    configs=[
        triton.Config(
            {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": gm}, num_warps=w, num_stages=st
        )
        for bm, bn, bk, gm, w, st in [
            (32, 32, 128, 1, 4, 2),
            (32, 64, 128, 1, 4, 2),
            (64, 64, 128, 1, 4, 3),
            (64, 128, 128, 1, 4, 3),
            (128, 64, 128, 1, 4, 3),
            (64, 128, 256, 1, 8, 2),
            (64, 128, 128, 8, 4, 3),
            (128, 64, 128, 8, 4, 3),
            # M=4096 search leaders: Q/O and the large FFN projections
            # need larger M tiles; KV benefits from a wider K tile.
            (128, 128, 128, 1, 8, 3),
            (128, 128, 128, 8, 8, 3),
            (256, 128, 128, 1, 8, 2),
            (256, 128, 128, 4, 8, 2),
            (256, 128, 128, 8, 8, 2),
            (256, 128, 128, 16, 8, 2),
            (128, 64, 256, 1, 4, 3),
            (128, 64, 256, 4, 4, 3),
            (128, 64, 256, 8, 4, 3),
            (128, 64, 256, 16, 4, 3),
            (128, 64, 256, 1, 8, 3),
            (128, 64, 256, 8, 8, 3),
            (256, 128, 128, 128, 8, 2),
            (256, 128, 128, 64, 8, 2),
            (128, 128, 128, 64, 4, 4),
            (128, 128, 128, 32, 4, 4),
            (128, 128, 128, 1, 4, 4),
            (128, 256, 128, 16, 8, 3),
            (128, 256, 128, 32, 8, 3),
            (32, 32, 128, 1, 4, 1),
            (32, 64, 128, 1, 4, 1),
        ]
    ]
    + _SMALL_M_CONFIGS,
    key=["M", "N", "K", "SPLIT_K", "ROW_PTS"],
    prune_configs_by={"early_config_prune": _prune_w4a4},
    do_bench=autotune_do_bench,
)
@triton.jit
def _w4a4(
    A,
    AS,
    AG,
    B,
    BS,
    PTS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    first = (pid // (GROUP_M * nn)) * GROUP_M
    gm = tl.minimum(nm - first, GROUP_M)
    m = (first + pid % gm) * BM + tl.arange(0, BM)
    n = ((pid % (GROUP_M * nn)) // gm) * BN + tl.arange(0, BN)
    kp = tl.arange(0, BK // 2)
    sp = tl.arange(0, BK // 16)
    acc = tl.zeros((BM, BN), tl.float32)
    chunk: tl.constexpr = tl.cdiv(K, BK * SPLIT_K)
    for off in range(chunk):
        block = split * chunk + off
        k = block * (BK // 2) + kp
        s = block * (BK // 16) + sp
        a = tl.load(
            A + m[:, None] * (K // 2) + k[None, :],
            (m[:, None] < M) & (k[None, :] < K // 2),
            0,
        )
        b = tl.load(
            B + n[None, :] * (K // 2) + k[:, None],
            (n[None, :] < N) & (k[:, None] < K // 2),
            0,
        )
        sa = tl.load(
            AS + m[:, None] * (K // 16) + s[None, :],
            (m[:, None] < M) & (s[None, :] < K // 16),
            0,
        ).to(tl.float8e4nv, bitcast=True)
        sb = tl.load(
            BS + n[:, None] * (K // 16) + s[None, :],
            (n[:, None] < N) & (s[None, :] < K // 16),
            0,
        ).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
    g = tl.load(AG + m, m < M, 0)
    if ROW_PTS:
        pts = tl.load(PTS + n, n < N, 0)[None, :]
    else:
        pts = tl.load(PTS)
    value = acc if SPLIT_K > 1 else acc * g[:, None] * pts
    tl.store(
        C + split.to(tl.int64) * M * N + m[:, None] * N + n[None, :],
        value,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _partial_offsets(split, index, M, N: tl.constexpr):
    plane = tl.cast(M, tl.int64) * N
    return tl.cast(split, tl.int64) * plane + tl.cast(index, tl.int64), plane


@triton.jit
def _reduce_partials(
    P,
    G,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FP4_ACT: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    offsets, plane = _partial_offsets(splits[:, None], index[None, :], M, N)
    partial = tl.load(P + offsets, index[None, :] < plane, 0)
    value = tl.sum(partial, 0)
    if FP4_ACT:
        if ROW_PTS:
            pts = tl.load(PTS + index % N, index < plane, 0)
        else:
            pts = tl.load(PTS)
        value *= tl.load(G + index // N, index < plane, 0) * pts
    tl.store(Y + index, value, index < plane)


def _launch_w4a4_impl(
    x, qdata, scale, pts, static_scale, *, split_k, output_dtype=None
):
    m, k = x.shape
    n = qdata.shape[0]
    _check_split(k, split_k, m=m, n=n)
    a, sa, g = nvfp4_quantize_activation(x, static_scale)
    out = torch.empty((m, n), device=x.device, dtype=output_dtype or x.dtype)
    partial = (
        torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        if split_k > 1
        else out
    )
    grid = lambda meta: (
        triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),
        split_k,
    )
    launch_nvfp4_kernel(
        _w4a4,
        grid,
        (a, sa, g, qdata, scale, pts, partial, m, n, k, split_k),
        7,
        ROW_PTS=pts.numel() != 1,
    )
    if split_k > 1 and not _QUEUING_ONLY.get():
        wrap_triton(_reduce_partials)[(triton.cdiv(m * n, 256),)](
            partial,
            g,
            pts,
            out,
            m,
            n,
            split_k,
            True,
            256,
            ROW_PTS=pts.numel() != 1,
        )
    return out


_launch_w4a4 = autotune_launch_param(
    "split_k",
    SPLIT_K_CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)(_launch_w4a4_impl)


@triton_op("triton::nvfp4_w4a4", mutates_args={})
def nvfp4_w4a4(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    static_scale: float = 0.0,
    split_k: int = -1,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    launch = _launch_w4a4 if split_k < 0 else _launch_w4a4_impl
    kwargs = {} if split_k < 0 else {"split_k": split_k}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    return launch(x, qdata, scale, per_tensor_scale, static_scale, **kwargs)


