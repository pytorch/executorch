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
from torch._subclasses.fake_tensor import is_fake
from torch.library import triton_op, wrap_triton
from torch.utils._sympy.numbers import int_oo

SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 32)


@torch._dynamo.assume_constant_result
def _split_workspace_limit():
    return (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory // 2
    )


def _max_rows(m):
    """M, or the upper bound of a dynamic M (None when unbounded)."""
    if m is None or isinstance(m, int):
        return m
    upper = m.node.shape_env.bound_sympy(m.node.expr).upper
    return None if upper == int_oo else int(upper)


def _check_split_workspace(split_k, max_rows, n):
    # The (split_k, M, N) workspace is allocated at the runtime M, so a split
    # must fit M's upper bound. The ops compute the bound from the traced M and
    # pass it to the autotuned launch, so timing (which sees hinted sizes) and
    # the traced call reject the same splits; None means unbounded.
    if split_k > 1:
        rows = _max_rows(max_rows)
        if rows is None:
            raise InvalidLaunchParam(f"NVFP4 split-K={split_k} needs a bounded M")
        workspace = split_k * rows * _size_hint(n) * 4
        if workspace > _split_workspace_limit():
            raise InvalidLaunchParam(
                f"NVFP4 split-K={split_k} needs {workspace} workspace bytes; "
                "the autotune limit is half the device memory"
            )


def _check_split(k, split_k, *, m=None, n=None):
    """``m`` is M or its upper bound (None: unbounded); the workspace is only
    checked when ``n`` is given."""
    if split_k not in SPLIT_K_CANDIDATES or split_k > triton.cdiv(k, 128):
        raise InvalidLaunchParam(f"Illegal NVFP4 split-K={split_k} for K={k}")
    if n is not None:
        _check_split_workspace(split_k, m, n)


def _check_weight(qdata, scale, per_tensor_scale):
    n, half_k = qdata.shape
    if half_k % 8 or scale.shape != (n, half_k // 8):
        raise ValueError(
            f"NVFP4 scale {tuple(scale.shape)} must be (N, K/16) for qdata "
            f"{tuple(qdata.shape)}"
        )
    if per_tensor_scale.numel() not in (1, n):
        raise ValueError("NVFP4 per_tensor_scale must have 1 or N values")
    if (qdata.dtype, scale.dtype, per_tensor_scale.dtype) != (
        torch.uint8,
        torch.uint8,
        torch.float32,
    ):
        raise ValueError("NVFP4 qdata/scale must be uint8, per_tensor_scale float32")
    if not all(t.is_contiguous() for t in (qdata, scale, per_tensor_scale)):
        raise ValueError("NVFP4 weight tensors must be contiguous")
    if scale.device != qdata.device or per_tensor_scale.device != qdata.device:
        raise ValueError("NVFP4 weight tensors must be on one device")


def _check_static_scale(static_scale, x):
    if static_scale is None:
        return
    if (
        static_scale.numel() != 1
        or static_scale.dtype != torch.float32
        or static_scale.device != x.device
    ):
        raise ValueError(
            "NVFP4 static activation scale must be a one-element float32 tensor "
            "on the activation's device"
        )
    # Only eager calls have values to check: fake tensors carry none, and a
    # CUDA graph capture cannot synchronize.
    if (
        not is_fake(static_scale)
        and static_scale.device.type != "meta"
        and not (static_scale.is_cuda and torch.cuda.is_current_stream_capturing())
        and not bool((torch.isfinite(static_scale) & (static_scale > 0)).all())
    ):
        raise ValueError("NVFP4 static activation scale must be finite and positive")


def _check_operands(x, qdata, scale, per_tensor_scale):
    _check_weight(qdata, scale, per_tensor_scale)
    k = x.shape[-1]
    if qdata.shape[-1] * 2 != k:
        raise ValueError(
            f"NVFP4 weight K ({qdata.shape[-1] * 2}) must match activation K ({k})"
        )
    if not x.is_contiguous():
        raise ValueError("NVFP4 activation must be contiguous")
    if x.device != qdata.device:
        raise ValueError("NVFP4 activation and weight must be on one device")


@triton.jit
def _fp4(code):
    mag = code & 7
    val = tl.where(
        mag < 2,
        mag.to(tl.float32) * 0.5,
        (((mag >> 1).to(tl.int32) + 126) << 23).to(tl.float32, bitcast=True)
        * (1.0 + (mag & 1).to(tl.float32) * 0.5),
    )
    # OR the sign bit in: float negation of +0 does not yield -0 here.
    sign = (code & 8).to(tl.uint32) << 28
    return (val.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _row_scale(
    X, G, STATIC, M, K: tl.constexpr, HAS_STATIC: tl.constexpr, BK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    if HAS_STATIC:
        g = tl.load(STATIC)
    else:
        k = tl.arange(0, BK)
        x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
        g = tl.max(tl.abs(x), 0) / (448.0 * 6.0)
    tl.store(G + row, tl.where(g > 0, g, 1.0))


@triton.jit
def _activation(X, G, Q, S, M, K: tl.constexpr, BG: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
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
def _has_native_e4m3():
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (8, 9)
    )


@triton.jit
def _e4m3_to_f32(b, NATIVE_E4M3: tl.constexpr):
    """E4M3 bytes to FP32: native on SM89+, else built bit-exactly in integer
    math (no FP32 denormal arithmetic); 0x7F/0xFF decode to NaN either way."""
    if NATIVE_E4M3:
        return b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        u = b.to(tl.int32)
        e = (u >> 3) & 15
        normal = (((u & 0x7F) << 20) + (120 << 23)).to(tl.float32, bitcast=True)
        mag = tl.where(e == 0, (u & 7).to(tl.float32) * 0.001953125, normal)
        nan = tl.full(mag.shape, 0x7FC00000, tl.int32).to(tl.float32, bitcast=True)
        mag = tl.where((u & 0x7F) == 0x7F, nan, mag)
        # OR the sign bit in: float negation of +0 does not yield -0 here.
        sign = (u & 0x80) << 24
        return (mag.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@triton.jit
def _fp4_pair_f16(q):
    """Byte of two E2M1 codes -> (low, high) FP32 values times 2**-14, exact."""
    u = q.to(tl.uint32)
    lo = (((u & 8) << 12) | ((u & 7) << 9)).to(tl.uint16)
    hi = (((u & 0x80) << 8) | ((u & 0x70) << 5)).to(tl.uint16)
    return (
        lo.to(tl.float16, bitcast=True).to(tl.float32),
        hi.to(tl.float16, bitcast=True).to(tl.float32),
    )


@triton.jit
def _e4m3_f16(b):
    """Finite E4M3 byte -> FP32 value times 2**-8, exact (FP16 holds every
    finite E4M3 value). ``CudaNVFP4Tensor`` scales are finite (NaN bytes are
    rejected at packing), so no NaN select runs in the hot loops."""
    u = b.to(tl.uint32)
    bits = (((u & 0x80) << 8) | ((u & 0x7F) << 7)).to(tl.uint16)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


# 2**14 (E2M1 FP16 placement) * 2**8 (E4M3 FP16 placement).
_SOFT_FP4_RESCALE = tl.constexpr(4194304.0)


@triton.jit
def _decode_pair(q, s, NATIVE: tl.constexpr):
    """Byte of two E2M1 codes and its E4M3 scale byte -> the exact FP32
    products (low * scale, high * scale): native conversion on SM120, FP16 bit
    placement elsewhere."""
    if NATIVE:
        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 tmp,z1,z2,z3; mov.b32 {tmp,z1,z2,z3}, $1; cvt.rn.f16x2.e2m1x2 $0, tmp; }",
            constraints="=r,r",
            args=[q.to(tl.uint32)],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        lo = (packed & 65535).to(tl.uint16).to(tl.float16, bitcast=True)
        hi = (packed >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
        sf = s.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        return lo.to(tl.float32) * sf, hi.to(tl.float32) * sf
    else:
        lo, hi = _fp4_pair_f16(q)
        sf = _e4m3_f16(s) * _SOFT_FP4_RESCALE
        return lo * sf, hi * sf


@torch._dynamo.assume_constant_result
def _has_native_fp4_conversion():
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 12
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (4, 8, 16)],
    key=["M", "K", "HAS_STATIC"],
    do_bench=autotune_do_bench,
)
@triton.jit
def _fused_activation(
    X,
    Q,
    S,
    G,
    STATIC,
    M,
    K: tl.constexpr,
    HAS_STATIC: tl.constexpr,
    BK: tl.constexpr,
):
    # One row load supplies both global amax and K16 scales. The native
    # conversion packs two E2M1 values in one instruction, low nibble first.
    row = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, BK // 16)
    idx = group[:, None] * 16 + tl.arange(0, 16)[None, :]
    x = tl.load(X + row * K + idx, idx < K, 0).to(tl.float32)
    block_max = tl.max(tl.abs(x), 1)
    if HAS_STATIC:
        g = tl.load(STATIC)
    else:
        g = tl.max(block_max, 0) / (448.0 * 6.0)
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
    x: torch.Tensor, static_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``static_scale``: an optional one-element float32 activation global
    scale, read by the kernels at run time (``None``: per-row dynamic)."""
    if x.shape[1] % 16:
        raise ValueError("NVFP4 activation K must be divisible by 16")
    if not x.is_contiguous():
        raise ValueError("NVFP4 activation must be contiguous")
    _check_static_scale(static_scale, x)
    return _quantize_activation(x, static_scale)


def _quantize_activation(x, static_scale):
    # Unchecked: the public ops validate once at entry, and launch timing
    # calls this on random stand-ins whose values mean nothing.
    m, k = x.shape
    q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty((m, k // 16), device=x.device, dtype=torch.uint8)
    g = torch.empty((m,), device=x.device, dtype=torch.float32)
    has_static = static_scale is not None
    static = static_scale if has_static else g
    if _has_native_fp4_conversion():
        launch_nvfp4_kernel(
            _fused_activation,
            (m,),
            (x, q, s, g, static, m, k, has_static, triton.next_power_of_2(k)),
            5,
        )
    else:
        wrap_triton(_row_scale)[(m,)](
            x, g, static, m, k, has_static, triton.next_power_of_2(k)
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
    m = ((first + pid % gm) * BM + tl.arange(0, BM)).to(tl.int64)
    n = (((pid % (GROUP_M * nn)) // gm) * BN + tl.arange(0, BN)).to(tl.int64)
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
    offsets, _ = _partial_offsets(split, m[:, None] * N + n[None, :], M, N)
    tl.store(C + offsets, value, (m[:, None] < M) & (n[None, :] < N))


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
    x, qdata, scale, pts, static_scale, max_rows, *, split_k, output_dtype=None
):
    m, k = x.shape
    n = qdata.shape[0]
    _check_split(k, split_k, m=max_rows, n=n)
    a, sa, g = _quantize_activation(x, static_scale)
    out = torch.empty((m, n), device=x.device, dtype=output_dtype or x.dtype)
    partial = (
        torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        if split_k > 1
        else out
    )

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]), split_k)

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
    static_scale: torch.Tensor | None = None,
    split_k: int = -1,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _check_operands(x, qdata, scale, per_tensor_scale)
    _check_static_scale(static_scale, x)
    if not _has_native_fp4_conversion():
        raise RuntimeError("Native NVFP4 W4A4 requires an SM120 or newer CUDA target")
    launch = _launch_w4a4 if split_k < 0 else _launch_w4a4_impl
    kwargs = {} if split_k < 0 else {"split_k": split_k}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    return launch(
        x,
        qdata,
        scale,
        per_tensor_scale,
        static_scale,
        _max_rows(x.shape[0]),
        **kwargs,
    )


@triton.jit
def _scaled_pair(q, sf, HALF2: tl.constexpr):
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp,z1,z2,z3; mov.b32 {tmp,z1,z2,z3}, $1; cvt.rn.f16x2.e2m1x2 $0, tmp; }",
        constraints="=r,r",
        args=[q.to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    if HALF2:
        # E2M1*E4M3 needs <=6 significant bits. The nonzero magnitude
        # range [2**-10, 2688] is normal FP16, so packed multiply is exact.
        sf_bits = sf.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        sf_pair = sf_bits | (sf_bits << 16)
        packed = tl.inline_asm_elementwise(
            "mul.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[packed, sf_pair],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    lo = (packed & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    hi = (packed >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    if not HALF2:
        lo = lo * sf
        hi = hi * sf
    return lo, hi


@triton.jit
def _gemv_half2(
    X,
    Q,
    S,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
    UNIT_SCALE: tl.constexpr = False,
):
    n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    m = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2)
    chunk: tl.constexpr = tl.cdiv(K, BK * SPLIT) * BK
    pair = split * (chunk // 2) + tl.arange(0, BK // 2)
    acc = tl.zeros((BN, BK // 2), tl.float32)
    if UNIT_SCALE:
        pts = 1.0
    elif ROW_PTS:
        pts = tl.load(PTS + n, n < N, 0)[:, None]
    else:
        pts = tl.load(PTS)
    for off in range(tl.cdiv(chunk, BK)):
        kk = pair + off * (BK // 2)
        mask = (n[:, None] < N) & (kk[None, :] < K // 2)
        q = tl.load(Q + n[:, None] * (K // 2) + kk[None, :], mask, 0)
        # One scale load per K16 group, broadcast to its eight byte pairs.
        group = split * (chunk // 16) + off * (BK // 16) + tl.arange(0, BK // 16)
        sf = (
            tl.load(
                S + n[:, None] * (K // 16) + group[None, :],
                (n[:, None] < N) & (group[None, :] < K // 16),
                0,
            )
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float16)
        )
        sf = tl.broadcast_to(sf[:, :, None], (BN, BK // 16, 8)).reshape((BN, BK // 2))
        lo, hi = _scaled_pair(q, sf, True)
        if UNIT_SCALE and X.dtype.element_ty == tl.bfloat16:
            w0, w1 = lo, hi
        elif X.dtype.element_ty == tl.bfloat16:
            packed_bf16 = tl.inline_asm_elementwise(
                "cvt.rn.bf16x2.f32 $0, $2, $1;",
                constraints="=r,f,f",
                args=[lo * pts, hi * pts],
                dtype=tl.uint32,
                is_pure=True,
                pack=1,
            )
            w0 = (packed_bf16 << 16).to(tl.float32, bitcast=True)
            w1 = (packed_bf16 & 0xFFFF0000).to(tl.float32, bitcast=True)
        else:
            w0 = (lo * pts).to(X.dtype.element_ty).to(tl.float32)
            w1 = (hi * pts).to(X.dtype.element_ty).to(tl.float32)
        x0 = tl.load(X + m * K + kk * 2, kk < K // 2, 0).to(tl.float32)
        x1 = tl.load(X + m * K + kk * 2 + 1, kk < K // 2, 0).to(tl.float32)
        acc = tl.fma(w0, x0[None, :], acc)
        acc = tl.fma(w1, x1[None, :], acc)
    val = tl.sum(acc, 1)
    offsets, _ = _partial_offsets(split, m.to(tl.int64) * N + n, M, N)
    tl.store(Y + offsets, val, n < N)


@triton.jit
def _gemv_original(
    X,
    Q,
    S,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NATIVE: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
    UNIT_SCALE: tl.constexpr = False,
):
    n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    m = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2)
    chunk: tl.constexpr = tl.cdiv(K, BK * SPLIT) * BK
    pair = split * (chunk // 2) + tl.arange(0, BK // 2)
    acc = tl.zeros((BN, BK // 2), tl.float32)
    if UNIT_SCALE:
        pts = 1.0
    elif ROW_PTS:
        pts = tl.load(PTS + n, n < N, 0)[:, None]
    else:
        pts = tl.load(PTS)
    for off in range(tl.cdiv(chunk, BK)):
        kk = pair + off * (BK // 2)
        mask = (n[:, None] < N) & (kk[None, :] < K // 2)
        q = tl.load(Q + n[:, None] * (K // 2) + kk[None, :], mask, 0)
        lo, hi = _decode_pair(
            q, tl.load(S + n[:, None] * (K // 16) + kk[None, :] // 8, mask, 0), NATIVE
        )
        # Unit-scale E2M1*E4M3 products have <=6 significant bits: exact in BF16.
        if UNIT_SCALE and X.dtype.element_ty == tl.bfloat16:
            w0, w1 = lo, hi
        else:
            w0 = (lo * pts).to(X.dtype.element_ty).to(tl.float32)
            w1 = (hi * pts).to(X.dtype.element_ty).to(tl.float32)
        x0 = tl.load(X + m * K + kk * 2, kk < K // 2, 0).to(tl.float32)
        x1 = tl.load(X + m * K + kk * 2 + 1, kk < K // 2, 0).to(tl.float32)
        acc += w0 * x0[None, :] + w1 * x1[None, :]
    val = tl.sum(acc, 1)
    offsets, _ = _partial_offsets(split, m.to(tl.int64) * N + n, M, N)
    tl.store(Y + offsets, val, n < N)


@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk, "HALF2": half}, num_warps=w, num_stages=st)
        for bn, bk, w in [
            (4, 256, 8),
            (4, 512, 4),
            (8, 256, 4),
            (8, 512, 4),
            (8, 1024, 8),
            (16, 256, 4),
            (16, 512, 4),
            (16, 512, 8),
            (16, 1024, 8),
            (32, 256, 4),
        ]
        for half in (False, True)
        for st in (1, 2, 3)
    ],
    key=["M", "N", "K", "SPLIT", "NATIVE", "ROW_PTS", "UNIT_SCALE"],
    prune_configs_by={"early_config_prune": prune_nvfp4_stages},
    do_bench=autotune_do_bench,
)
@triton.jit
def _gemv(
    X,
    Q,
    S,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NATIVE: tl.constexpr,
    HALF2: tl.constexpr = False,
    ROW_PTS: tl.constexpr = False,
    UNIT_SCALE: tl.constexpr = False,
):
    if NATIVE and HALF2:
        _gemv_half2(X, Q, S, PTS, Y, M, N, K, SPLIT, BN, BK, ROW_PTS, UNIT_SCALE)
    else:
        _gemv_original(
            X,
            Q,
            S,
            PTS,
            Y,
            M,
            N,
            K,
            SPLIT,
            BN,
            BK,
            NATIVE,
            ROW_PTS,
            UNIT_SCALE,
        )


def _launch_gemv_impl(x, qdata, scale, pts, max_rows, unit_scale=False, *, split_k):
    m, k = x.shape
    n = qdata.shape[0]
    if split_k not in SPLIT_K_CANDIDATES:
        raise InvalidLaunchParam(f"Illegal NVFP4 GEMV split-K={split_k}")
    _check_split_workspace(split_k, max_rows, n)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    partial = (
        torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        if split_k > 1
        else out
    )

    def grid(meta):
        return (triton.cdiv(n, meta["BN"]), m, split_k)

    launch_nvfp4_kernel(
        _gemv,
        grid,
        (
            x,
            qdata,
            scale,
            pts,
            partial,
            m,
            n,
            k,
            split_k,
        ),
        5,
        NATIVE=_has_native_fp4_conversion(),
        ROW_PTS=pts.numel() != 1,
        UNIT_SCALE=unit_scale,
    )
    if split_k > 1 and not _QUEUING_ONLY.get():
        wrap_triton(_reduce_partials)[(triton.cdiv(m * n, 256),)](
            partial,
            pts,
            pts,
            out,
            m,
            n,
            split_k,
            False,
            256,
            ROW_PTS=False,
        )
    return out


_launch_gemv = autotune_launch_param(
    "split_k",
    SPLIT_K_CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)(_launch_gemv_impl)


@triton_op("triton::nvfp4_gemv", mutates_args={})
def nvfp4_gemv(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    split_k: int = -1,
    unit_scale: bool = False,
) -> torch.Tensor:
    """NVFP4 GEMV for small M; the output keeps ``x.dtype``.

    Scale bytes must be finite E4M3 (no 0x7F/0xFF), as ``CudaNVFP4Tensor``
    guarantees; it is not checked here. ``unit_scale=True`` skips the global
    scale and requires the ``CudaNVFP4Tensor.unit_scale`` contract (global
    scale 1). M is grid axis 1, so at most 65535.
    """
    _check_operands(x, qdata, scale, per_tensor_scale)
    max_rows = _max_rows(x.shape[0])
    if max_rows is None or max_rows > 65535:
        raise ValueError("NVFP4 GEMV supports at most 65535 rows")
    args = (x, qdata, scale, per_tensor_scale, max_rows, unit_scale)
    if split_k < 0:
        return _launch_gemv(*args)
    return _launch_gemv_impl(*args, split_k=split_k)


@triton.autotune(
    configs=[
        triton.Config(
            {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": gm}, num_warps=w, num_stages=st
        )
        for bm, bn, bk, gm, w in [
            (16, 64, 64, 1, 4),
            (32, 64, 64, 1, 4),
            (64, 64, 64, 1, 4),
            (64, 128, 64, 1, 8),
            (32, 64, 128, 1, 4),
            (64, 64, 128, 1, 4),
            (64, 128, 64, 8, 8),
            (128, 64, 64, 1, 4),
            (128, 64, 64, 8, 4),
            (128, 64, 64, 8, 8),
            (128, 64, 128, 8, 8),
            (128, 64, 128, 1, 8),
        ]
        for st in (1, 2, 3)
    ],
    key=["M", "N", "K", "SPLIT_K", "NATIVE", "ROW_PTS"],
    prune_configs_by={"early_config_prune": prune_nvfp4_stages},
    do_bench=autotune_do_bench,
)
@triton.jit
def _w4a16(
    X,
    Q,
    S,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    NATIVE: tl.constexpr = False,
    ROW_PTS: tl.constexpr = False,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    first = (pid // (GROUP_M * nn)) * GROUP_M
    gm = tl.minimum(nm - first, GROUP_M)
    m = ((first + pid % gm) * BM + tl.arange(0, BM)).to(tl.int64)
    n = (((pid % (GROUP_M * nn)) // gm) * BN + tl.arange(0, BN)).to(tl.int64)
    ki = tl.arange(0, BK)
    pi = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    if ROW_PTS:
        pts = tl.load(PTS + n, n < N, 0)[:, None]
    else:
        pts = tl.load(PTS)
    chunk: tl.constexpr = tl.cdiv(K, BK * SPLIT_K)
    for off in range(chunk):
        block = split * chunk + off
        kk = ki + block * BK
        pair = pi + block * (BK // 2)
        a = tl.load(
            X + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0
        )
        mask = (n[:, None] < N) & (pair[None, :] < K // 2)
        q = tl.load(Q + n[:, None] * (K // 2) + pair[None, :], mask, 0)
        lo, hi = _decode_pair(
            q, tl.load(S + n[:, None] * (K // 16) + pair[None, :] // 8, mask, 0), NATIVE
        )
        w0 = (lo * pts).to(X.dtype.element_ty)
        w1 = (hi * pts).to(X.dtype.element_ty)
        b = tl.trans(tl.join(w0, w1).reshape((BN, BK)))
        acc = tl.dot(a, b, acc)
    tl.store(
        Y + split.to(tl.int64) * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


def _launch_w4a16_impl(x, qdata, scale, pts, max_rows, *, split_k, output_dtype=None):
    m, k = x.shape
    n = qdata.shape[0]
    _check_split(k, split_k, m=max_rows, n=n)
    output_dtype = output_dtype or x.dtype
    if x.dtype == torch.float32:
        x = x.to(torch.bfloat16)
    out = torch.empty((m, n), device=x.device, dtype=output_dtype)
    partial = (
        torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        if split_k > 1
        else out
    )

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]), split_k)

    launch_nvfp4_kernel(
        _w4a16,
        grid,
        (x, qdata, scale, pts, partial, m, n, k, split_k),
        5,
        NATIVE=_has_native_fp4_conversion(),
        ROW_PTS=pts.numel() != 1,
    )
    if split_k > 1 and not _QUEUING_ONLY.get():
        wrap_triton(_reduce_partials)[(triton.cdiv(m * n, 256),)](
            partial,
            pts,
            pts,
            out,
            m,
            n,
            split_k,
            False,
            256,
            ROW_PTS=False,
        )
    return out


_launch_w4a16 = autotune_launch_param(
    "split_k",
    SPLIT_K_CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)(_launch_w4a16_impl)


@triton_op("triton::nvfp4_w4a16", mutates_args={})
def nvfp4_w4a16(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    split_k: int = -1,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """NVFP4 weight-only GEMM; FP32 activations are rounded to BF16 (the GEMV
    keeps FP32 math), and the output is ``output_dtype`` or ``x.dtype``. Scale
    bytes must be finite E4M3 (no 0x7F/0xFF), as ``CudaNVFP4Tensor``
    guarantees; it is not checked here."""
    _check_operands(x, qdata, scale, per_tensor_scale)
    launch = _launch_w4a16 if split_k < 0 else _launch_w4a16_impl
    kwargs = {} if split_k < 0 else {"split_k": split_k}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    return launch(x, qdata, scale, per_tensor_scale, _max_rows(x.shape[0]), **kwargs)


@triton.jit
def _embedding(
    I,
    Q,
    S,
    PTS,
    Y,
    V,
    K: tl.constexpr,
    BK: tl.constexpr,
    ROW_PTS: tl.constexpr = False,
    NATIVE_E4M3: tl.constexpr = False,
):
    t = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    n = tl.load(I + t).to(tl.int64)
    # Out-of-range ids are rejected on the host in eager mode; never read
    # outside the table for them.
    valid = (n >= 0) & (n < V)
    n = tl.where(valid, n, 0)
    mask = (k < K) & valid
    q = tl.load(Q + n * (K // 2) + k // 2, mask, 0)
    c = (q >> ((k & 1) * 4)) & 15
    s = _e4m3_to_f32(tl.load(S + n * (K // 16) + k // 16, mask, 0), NATIVE_E4M3)
    pts = tl.load(PTS + n) if ROW_PTS else tl.load(PTS)
    val = _fp4(c) * (s * pts)
    tl.store(Y + t * K + k, val, k < K)


@triton_op("triton::nvfp4_embedding", mutates_args={})
def nvfp4_embedding(
    indices: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Eager calls outside CUDA-graph capture reject out-of-range ids; compiled
    graphs and captured calls do not validate them and return zero rows."""
    _check_weight(qdata, scale, per_tensor_scale)
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("NVFP4 embedding ids must be int32 or int64")
    if indices.device != qdata.device:
        raise ValueError("NVFP4 embedding ids and weight must be on one device")
    v, k = qdata.shape[0], qdata.shape[1] * 2
    if (
        not is_fake(indices)
        and indices.numel()
        and not torch.cuda.is_current_stream_capturing()
        and bool(((indices < 0) | (indices >= v)).any())
    ):
        raise IndexError(f"NVFP4 embedding index out of range for {v} rows")
    indices = indices.contiguous()
    out = torch.empty((*indices.shape, k), device=qdata.device, dtype=output_dtype)
    wrap_triton(_embedding)[(indices.numel(), triton.cdiv(k, 256))](
        indices,
        qdata,
        scale,
        per_tensor_scale,
        out,
        v,
        k,
        256,
        ROW_PTS=per_tensor_scale.numel() != 1,
        NATIVE_E4M3=_has_native_e4m3(),
    )
    return out
