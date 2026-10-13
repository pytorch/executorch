# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pieces shared by the decode-sized quantized GEMM kernels (INT4/5/6/8).

* ``@triton.jit`` helpers (DP4A, warp sum, round-to-nearest-even): inline PTX
  on CUDA, portable Triton on ROCm;
* BF16 -> signed INT8 activation quantization in K32 blocks, the
  W*A8 formats' first kernel, and its launcher;
* the deterministic split-K reduce kernel, the split-K candidates the launch
  functions time during AOTI compile, and their legality check;
* generic row and tensor-core tile autotune spaces and their pruning;
* legality checks the formats compose into their ``unsupported_reason``.

Nothing here is tuned per architecture or per shape.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.autotune.launch_params import InvalidLaunchParam
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import wrap_triton
from triton.language.extra import libdevice

# Fixed when this module is imported, never by querying the driver while a
# kernel compiles: Inductor compiles in forked subprocesses that cannot
# initialize CUDA. Inductor inlines the value into the kernel source it builds.
_IS_HIP = tl.constexpr(torch.version.hip is not None)

# Activation quantization granularity: one INT8 scale per K32 block, launched in
# K256 tiles.
Q8_BLOCK = 32
Q8_TILE = 256
_TL_Q8_BLOCK = tl.constexpr(Q8_BLOCK)
_TL_Q8_TILE = tl.constexpr(Q8_TILE)

# Split-K values each launch function times during AOTI compile
# (@autotune_launch_param); outside it, the first one is used.
SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16)
# Generic autotune spaces. num_stages and the kernel's PIPELINE_STAGES are
# always the same value.
ROWS_PER_CTA_CHOICES = (1, 2, 4, 8)
PIPELINE_STAGE_CHOICES = (1, 2, 3)
TILE_BLOCK_M_CHOICES = (8, 16, 32, 64)
TILE_BLOCK_N_CHOICES = (32, 64, 128)
TILE_WARP_CHOICES = (4, 8)


@triton.jit
def _round_nearest_even_s32_ptx(value):
    return tl.inline_asm_elementwise(
        asm="cvt.rni.s32.f32 $0, $1;",
        constraints="=r,f",
        args=[value],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dp4a_u8_s8_ptx(a, b, acc):
    return tl.inline_asm_elementwise(
        asm="dp4a.u32.s32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[a, b, acc],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _warp_sum_f32_ptx(value):
    shuffled = tl.inline_asm_elementwise(
        asm="shfl.sync.bfly.b32 $0, $1, 16, 0x1f, 0xffffffff;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value += shuffled
    shuffled = tl.inline_asm_elementwise(
        asm="shfl.sync.bfly.b32 $0, $1, 8, 0x1f, 0xffffffff;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value += shuffled
    shuffled = tl.inline_asm_elementwise(
        asm="shfl.sync.bfly.b32 $0, $1, 4, 0x1f, 0xffffffff;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value += shuffled
    shuffled = tl.inline_asm_elementwise(
        asm="shfl.sync.bfly.b32 $0, $1, 2, 0x1f, 0xffffffff;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value += shuffled
    shuffled = tl.inline_asm_elementwise(
        asm="shfl.sync.bfly.b32 $0, $1, 1, 0x1f, 0xffffffff;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return value + shuffled


# ROCm versions of the PTX helpers above. _IS_HIP is a compile-time constant,
# so CUDA builds see exactly the PTX helpers.


@triton.jit
def _round_nearest_even_s32_portable(value):
    return libdevice.rint(value).to(tl.int32)


@triton.jit
def _dp4a_u8_s8_portable(a, b, acc):
    """acc + sum over the four bytes of unsigned(a) * signed(b), like dp4a.u32.s32."""
    a = a.to(tl.uint32, bitcast=True)
    b = b.to(tl.int32, bitcast=True)
    acc += (a & 0xFF).to(tl.int32) * ((b << 24) >> 24)
    acc += ((a >> 8) & 0xFF).to(tl.int32) * ((b << 16) >> 24)
    acc += ((a >> 16) & 0xFF).to(tl.int32) * ((b << 8) >> 24)
    acc += (a >> 24).to(tl.int32) * (b >> 24)
    return acc


@triton.jit
def _warp_sum_f32_portable(value):
    """Sum of each run of 32 consecutive elements of a 1-D ``value``, broadcast
    back to all 32: what the PTX butterfly computes when element i is lane
    i % 32, without depending on the warp size or the layout."""
    n: tl.constexpr = value.shape[0]
    groups = tl.reshape(value, (n // 32, 32), can_reorder=False)
    sums = tl.broadcast_to(tl.sum(groups, axis=1)[:, None], (n // 32, 32))
    return tl.reshape(sums, (n,), can_reorder=False)


@triton.jit
def _round_nearest_even_s32(value):
    if _IS_HIP:
        return _round_nearest_even_s32_portable(value)
    else:
        return _round_nearest_even_s32_ptx(value)


@triton.jit
def _dp4a_u8_s8(a, b, acc):
    if _IS_HIP:
        return _dp4a_u8_s8_portable(a, b, acc)
    else:
        return _dp4a_u8_s8_ptx(a, b, acc)


@triton.jit
def _warp_sum_f32(value):
    if _IS_HIP:
        return _warp_sum_f32_portable(value)
    else:
        return _warp_sum_f32_ptx(value)


@triton.jit
def _quantize_activations_q8_kernel(
    x,
    qwords,
    x_scale,
    x_sum,
    M,
    K: tl.constexpr,
    STORE_SUM: tl.constexpr,
    NATURAL_ORDER: tl.constexpr,
):
    """Quantize one K256 tile into eight packed signed-Q8 K32 blocks."""
    super_block = tl.program_id(0)
    row = tl.program_id(1)
    valid_row = row < M
    groups: tl.constexpr = K // _TL_Q8_BLOCK

    offs_k = super_block * _TL_Q8_TILE + tl.arange(0, 256)
    values = tl.load(
        x + row * K + offs_k,
        mask=valid_row,
        other=0.0,
    ).to(tl.float32)
    values = tl.reshape(values, (8, 32), can_reorder=False)
    absmax = tl.max(tl.abs(values), axis=1)
    scale_value = absmax / 127.0
    inv_scale = tl.where(absmax > 0.0, 1.0 / scale_value, 0.0)
    scaled = tl.maximum(tl.minimum(values * inv_scale[:, None], 127.0), -127.0)
    quantized = _round_nearest_even_s32(scaled)
    quantized = tl.maximum(tl.minimum(quantized, 127), -128)
    if STORE_SUM:
        sum_value = tl.sum(quantized, axis=1)

    thread = tl.arange(0, 256)
    local_group = thread // 32
    lane = thread % 32
    group = super_block * 8 + local_group
    if NATURAL_ORDER:
        byte_in_group = lane
    else:
        byte_in_group = tl.where((lane & 1) == 0, lane // 2, 16 + lane // 2)
    quantized_bytes = tl.reshape(quantized, (256,), can_reorder=False) & 0xFF
    qbytes = qwords.to(tl.pointer_type(tl.uint8))
    tl.store(
        qbytes + (row * groups + group) * 32 + byte_in_group,
        quantized_bytes,
        mask=valid_row,
    )
    metadata_group = super_block * 8 + tl.arange(0, 8)
    tl.store(x_scale + row * groups + metadata_group, scale_value, mask=valid_row)
    if STORE_SUM:
        tl.store(x_sum + row * groups + metadata_group, sum_value, mask=valid_row)


@triton.jit
def _splitk_reduce_kernel(
    partial,
    out,
    M,
    N: tl.constexpr,
    stride_ps: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_id in range(SPLIT_K):
        acc += tl.load(
            partial
            + split_id * stride_ps
            + offs_m[:, None] * stride_pm
            + offs_n[None, :] * stride_pn,
            mask=mask,
            other=0.0,
        )
    tl.store(
        out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=mask,
    )


def quantize_activations_q8(
    x: torch.Tensor,
    store_sum: bool = True,
    natural_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantizes BF16 ``x`` [M, K] (K % 256 == 0) to signed INT8 per K32 block.

    Returns the packed INT8 values as int32 words [M, K/4], the FP32 block
    scales [M, K/32], and the INT32 block sums [M, K/32] (a 1-element
    placeholder when ``store_sum`` is False). The default byte layout groups
    even then odd K values for packed sub-byte weights; ``natural_order=True``
    stores each K32 block in increasing K order.
    """
    M, K = x.shape
    groups = K // Q8_BLOCK
    qwords = torch.empty((M, K // 4), dtype=torch.int32, device=x.device)
    x_scale = torch.empty((M, groups), dtype=torch.float32, device=x.device)
    x_sum = torch.empty(
        (M, groups) if store_sum else (1,), dtype=torch.int32, device=x.device
    )
    wrap_triton(_quantize_activations_q8_kernel)[(K // Q8_TILE, M)](
        x,
        qwords,
        x_scale,
        x_sum,
        M,
        K,
        STORE_SUM=store_sum,
        NATURAL_ORDER=natural_order,
        num_warps=8,
        num_stages=1,
    )
    return qwords, x_scale, x_sum


def splitk_reduce(
    partial: torch.Tensor, out: torch.Tensor, block_m: int, block_n: int = 64
) -> None:
    """Sums ``partial`` [SPLIT_K, >=M, N] in fixed order into BF16 ``out``."""
    M, N = out.shape
    wrap_triton(_splitk_reduce_kernel)[
        (triton.cdiv(N, block_n), triton.cdiv(M, block_m))
    ](
        partial,
        out,
        M,
        N,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLIT_K=partial.shape[0],
        num_warps=4,
        num_stages=1,
    )


def check_split_k(split_k: int, k: int, k_per_split_unit: int = Q8_TILE) -> None:
    """Raises ``InvalidLaunchParam`` unless each of the ``split_k`` K slices
    gets at least one ``k_per_split_unit``."""
    if not 1 <= split_k <= k // k_per_split_unit:
        raise InvalidLaunchParam(
            f"split_k={split_k} needs K >= {split_k * k_per_split_unit}, got K={k}"
        )


def autotune_configs(
    implementations: Iterable[dict] = ({},),
) -> list[triton.Config]:
    """The generic search space: each implementation (extra constexpr kwargs)
    x rows per CTA (BLOCK_N = num_warps, one output row per warp) x pipeline
    stages s (PIPELINE_STAGES = num_stages = s)."""
    return [
        triton.Config(
            {**implementation, "BLOCK_N": rows, "PIPELINE_STAGES": stages},
            num_warps=rows,
            num_stages=stages,
        )
        for implementation in implementations
        for rows in ROWS_PER_CTA_CHOICES
        for stages in PIPELINE_STAGE_CHOICES
    ]


def tile_autotune_configs(bucket: int) -> list[triton.Config]:
    """Generic tensor-core tile space for a static maximum-M bucket."""
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "PIPELINE_STAGES": stages,
            },
            num_warps=warps,
            num_stages=stages,
        )
        for block_m in TILE_BLOCK_M_CHOICES
        if block_m <= bucket
        for block_n in TILE_BLOCK_N_CHOICES
        for warps in TILE_WARP_CHOICES
        for stages in PIPELINE_STAGE_CHOICES
    ]


def prune_by_main_loop_trips(trips_per_split, configs, named_args, **kwargs):
    """``prune_configs_by`` early pruning: drop configs whose pipeline stage
    count exceeds the main loop's trip count, ``trips_per_split(args)``, where
    ``args`` holds the kernel's arguments by name. Keeps at least one config."""
    trips = trips_per_split({**named_args, **kwargs})
    kept = [c for c in configs if c.kwargs.get("PIPELINE_STAGES", 1) <= trips]
    return kept or [min(configs, key=lambda c: c.kwargs.get("PIPELINE_STAGES", 1))]


# ---------------------------------------------------------------------------
# Legality checks the formats compose into ``unsupported_reason``. Each returns
# None or a short reason, and never raises.
# ---------------------------------------------------------------------------


def check_dtypes(
    named: Sequence[tuple[str, torch.Tensor, Sequence[torch.dtype]]]
) -> Optional[str]:
    for name, tensor, dtypes in named:
        if tensor.dtype not in dtypes:
            allowed = "/".join(str(d).replace("torch.", "") for d in dtypes)
            return f"{name} must be {allowed}, got {str(tensor.dtype).replace('torch.', '')}"
    return None


def check_contiguous(tensors: Sequence[torch.Tensor]) -> Optional[str]:
    if not all(t.is_contiguous() for t in tensors):
        return "inputs must be contiguous"
    return None


def check_device(x: torch.Tensor, tensors: Sequence[torch.Tensor]) -> Optional[str]:
    from torch._subclasses.fake_tensor import is_fake

    if x.device.type != "cuda" and not is_fake(x):
        return "activation must be on a CUDA device (or fake while tracing)"
    if any(t.device != x.device for t in tensors):
        return "activation and weights must be on the same device"
    return None


def check_k(k, multiple: int) -> Optional[str]:
    if not isinstance(k, int):
        return "K must be static"
    if k % multiple != 0:
        return f"K must be a multiple of {multiple}, got {k}"
    return None


def check_rows(m, bucket: int) -> Optional[str]:
    """Static or dynamic M must lie in [1, bucket]."""
    if isinstance(m, int):
        return (
            None
            if 1 <= m <= bucket
            else f"static M must be within [1, {bucket}], got {m}"
        )
    if statically_known_true(m >= 1) and statically_known_true(m <= bucket):
        return None
    return f"dynamic M is not provably within [1, {bucket}]"


def first_reason(*reasons: Optional[str]) -> Optional[str]:
    return next((r for r in reasons if r is not None), None)
