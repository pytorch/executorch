#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Int4 affine (scale + zero point) weight linears lowered to Metal Performance
Primitives (Metal 4) kernels, with two activation modes:

* ``mlx::int8_int4_affine_gemm``: dynamic per-token affine int8 activations,
  INT8 x INT4 -> INT32 MPP accumulation per weight group.
* ``mlx::int8sym_int4_affine_gemm``: the same with symmetric per-token int8
  activations (scale only, as ``mlx::int8_intx_gemm``): no ``x_zero`` /
  ``w_sum`` term, same weights and GEMM kernels.
* ``mlx::fp_int4_affine_gemm``: fp16 / bf16 activations, half / bfloat x INT4
  -> FP32 MPP accumulation per weight group.

Weights are grouped along K: 64, 128 or 256, or ``GS == K`` for per-channel.
``wq`` is signed int4 packed as in
:func:`executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm.pack_int4`,
with a float scale and zero point per (channel, group) from
:func:`quantize_int4_affine`:

    W[n, k] = scale[n, g] * (wq[n, k] - zero[n, g]),  g = k // GS

The ops take that weight in bias form, as ``[G, N]`` planes from
:func:`int4_affine_planes`, plus its row sums:

    w_scale[g, n] = scale[n, g]
    w_bias[g, n]  = -scale[n, g] * zero[n, g]
    W[n, k]       = w_scale[g, n] * wq[n, k] + w_bias[g, n]
    w_sum[n]      = sum_k W[n, k]

Int8 activations use a per-token float scale and zero point over the token's
range (extended to include 0); ``round`` is round-half-to-even:

    lo, hi     = min(min_k x[m, k], 0), max(max_k x[m, k], 0)
    x_scale[m] = max((hi - lo) / 255, eps)
    x_zero[m]  = -128 - lo / x_scale[m]
    xq[m, k]   = clamp(round((x[m, k] - lo) * (1 / x_scale[m])), 0, 255) - 128
    x[m, k]   ~= x_scale[m] * (xq[m, k] - x_zero[m])

With ``D[m, n, g]`` the group's integer (or fp) dot product of activations and
``wq`` and ``S[m, g]`` the group's activation sum:

    F[m, n]       = sum_g w_scale[g, n] * D[m, n, g] + sum_g S[m, g] * w_bias[g, n]
    int8: Y[m, n] = x_scale[m] * (F[m, n] - x_zero[m] * w_sum[n])
    fp:   Y[m, n] = F[m, n]

``w_sum`` folds the activation zero point into one term per output; the fp op
does not use it. The bias form lets the weight zero point leave the per-group
loop (``sum_g S * w_bias`` is one small MPP matmul) and the ``[G, N]`` planes
give each thread 4 adjacent columns' scales per load.

Lowering mirrors ``mlx::int8_intx_gemm`` / ``mlx::fp_int4_gemm`` (matvec and
qmv_wide-style kernels for small M, MPP tiles otherwise, split-K plus a reduce
kernel when that adds enough threadgroups). The grouped MPP kernel follows the
M5 Max int8 x int4 group-wise study (digantdesai/m5_gemm_latest on Manifold,
int8_int4/EXPLORATION.md): one MPP op per SIMD group, a per-group rescale of
one fma per accumulator element using the measured accumulator layout, the
bias term as one MPP call or one pass after the group loop, and vector output
stores. A prologue kernel produces ``S`` (and, for int8, the quantized
activations). Per-channel int8 keeps exact INT32 sums through every kernel, so
its results are bit-identical across them as in ``mlx::int8_intx_gemm``.

Constraints for MLX lowering:
    * ``K`` and ``N`` must be statically known; ``K`` a multiple of 64.
    * ``GS`` must be a multiple of 64 that divides ``K``.
    * fp activations must be fp16 or bf16.
    * Requires an MPP-capable GPU (Metal 4).

Usage::

    import executorch.backends.mlx.custom_kernel_ops.int4_affine_gemm  # noqa: F401

    y = torch.ops.mlx.int8_int4_affine_gemm(x, w_packed, w_scale, w_bias, w_sum)
    y = torch.ops.mlx.fp_int4_affine_gemm(x, w_packed, w_scale, w_bias)
    # x:        (..., K)       fp16 / bf16 (/ fp32 for int8)
    # w_packed: (N, K // 2)    uint8
    # w_scale:  (K // GS, N)   float (one dtype for both planes)
    # w_bias:   (K // GS, N)   float
    # w_sum:    (N,)           float
    # y:        (..., N)       x.dtype
"""

from __future__ import annotations

import math
from string import Template
from typing import Callable, NamedTuple, Optional

import torch
from executorch.backends.mlx.custom_kernel_ops import fp_int4_gemm, int8_intx_gemm
from executorch.backends.mlx.custom_kernel_ops.int8_intx_gemm import (
    pack_int4,
    quantize_per_token_int8,
    unpack_int4,
)
from torch import nn, Tensor
from torch.fx.node import Node

_EPS = 1e-10
GROUP_SIZES = (64, 128, 256, None)
ACTS = ("int8", "int8sym", "fp")


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def quantize_per_token_int8_affine(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Kernel-matching affine per-token quantization of ``x [M, K]``.

    Returns ``(xq int8 [M, K], x_scale fp32 [M, 1], x_zero fp32 [M, 1])``.
    """
    xf = x.float()
    lo = xf.amin(dim=-1, keepdim=True).clamp(max=0)
    hi = xf.amax(dim=-1, keepdim=True).clamp(min=0)
    x_scale = ((hi - lo) / 255).clamp(min=_EPS)
    x_zero = -128 - lo / x_scale
    xq = torch.round((xf - lo) * torch.reciprocal(x_scale)).clamp(0, 255) - 128
    return xq.to(torch.int8), x_scale, x_zero


def quantize_int4_affine(
    rows: Callable[[int, int], Tensor],
    N: int,
    group_size: Optional[int],
    chunk_rows: int = 4096,
) -> tuple[Tensor, Tensor, Tensor]:
    """Affine int4 quantization of an ``[N, K]`` weight over groups of
    ``group_size`` along K (``None`` for per-channel). Each group's min / max
    map to -8 / 7 with a float zero point.

    ``rows(start, end)`` returns rows ``[start, end)`` as a float tensor, so a
    large weight is only ever materialized ``chunk_rows`` rows at a time.
    Returns ``(w_packed uint8 [N, K // 2], scale fp32 [N, G], zero fp32 [N, G])``.
    """
    packed, scales, zeros = [], [], []
    for start in range(0, N, chunk_rows):
        w = rows(start, min(start + chunk_rows, N)).float()
        n, K = w.shape
        gs = K if group_size is None else group_size
        if K % gs:
            raise ValueError(f"group_size {gs} does not divide K={K}")
        wg = w.view(n, K // gs, gs)
        lo = wg.amin(dim=-1)
        hi = wg.amax(dim=-1)
        scale = ((hi - lo) / 15).clamp(min=_EPS)
        wq = torch.round((wg - lo[..., None]) / scale[..., None]).clamp(0, 15) - 8
        packed.append(pack_int4(wq.view(n, K)))
        scales.append(scale)
        zeros.append(-8 - lo / scale)
        del w, wg, wq
    return torch.cat(packed), torch.cat(scales), torch.cat(zeros)


def int4_affine_planes(
    scale: Tensor, zero: Tensor, dtype: torch.dtype = torch.float16
) -> tuple[Tensor, Tensor]:
    """``[N, G]`` scale / zero point -> the ops' ``(w_scale, w_bias)``
    ``[G, N]`` planes in ``dtype``, with ``w_bias = -scale * zero``."""
    scale = scale.float()
    return (
        scale.T.contiguous().to(dtype),
        (-scale * zero.float()).T.contiguous().to(dtype),
    )


def dequantize_int4_affine(w_packed: Tensor, w_scale: Tensor, w_bias: Tensor) -> Tensor:
    """``fp32 [N, K]`` weight from ``w_packed`` and the ``[G, N]`` planes."""
    wq = unpack_int4(w_packed).float()
    N, K = wq.shape
    G = w_scale.shape[0]
    w = (
        wq.view(N, G, K // G) * w_scale.float().T[..., None]
        + w_bias.float().T[..., None]
    )
    return w.view(N, K)


def int4_affine_weight_sum(
    w_packed: Tensor, w_scale: Tensor, w_bias: Tensor, chunk_rows: int = 4096
) -> Tensor:
    """``w_sum fp32 [N]``: row sums of the dequantized weight (as the planes
    store it), dequantizing ``chunk_rows`` rows at a time."""
    sums = []
    for start in range(0, w_packed.shape[0], chunk_rows):
        end = min(start + chunk_rows, w_packed.shape[0])
        w = dequantize_int4_affine(
            w_packed[start:end], w_scale[:, start:end], w_bias[:, start:end]
        )
        sums.append(w.double().sum(-1).float())
    return torch.cat(sums)


# ---------------------------------------------------------------------------
# Custom ops + eager fallbacks
# ---------------------------------------------------------------------------


class Int4AffineLinear(nn.Module):
    """Bias-free linear computed by ``mlx::int8_int4_affine_gemm`` or
    ``mlx::fp_int4_affine_gemm`` depending on ``act``, from
    :func:`quantize_int4_affine`'s ``(w_packed, scale, zero)``. Stores the
    ops' ``[G, N]`` planes in ``qparam_dtype`` and ``w_sum`` in fp32."""

    def __init__(
        self,
        w_packed: Tensor,
        scale: Tensor,
        zero: Tensor,
        act: str,
        qparam_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if act not in ACTS:
            raise ValueError(f"act must be one of {ACTS}; got {act}")
        self.act = act
        self.in_features = 2 * w_packed.shape[1]
        self.out_features = w_packed.shape[0]
        self.group_size = self.in_features // scale.shape[1]
        w_scale, w_bias = int4_affine_planes(scale, zero, qparam_dtype)
        self.register_buffer("w_packed", w_packed)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("w_bias", w_bias)
        self.register_buffer("w_sum", int4_affine_weight_sum(w_packed, w_scale, w_bias))

    def _apply(self, fn, recurse=True):
        # Keep qparam dtypes through model.to(activation_dtype); move device only.
        qparams = (self.w_scale, self.w_bias, self.w_sum)
        result = super()._apply(fn, recurse)
        device = self.w_packed.device
        self.w_scale, self.w_bias, self.w_sum = (t.to(device) for t in qparams)
        return result

    def forward(self, x: Tensor) -> Tensor:
        if self.act == "int8":
            return torch.ops.mlx.int8_int4_affine_gemm(
                x, self.w_packed, self.w_scale, self.w_bias, self.w_sum
            )
        if self.act == "int8sym":
            return torch.ops.mlx.int8sym_int4_affine_gemm(
                x, self.w_packed, self.w_scale, self.w_bias
            )
        return torch.ops.mlx.fp_int4_affine_gemm(
            x, self.w_packed, self.w_scale, self.w_bias
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, act={self.act}"
        )


@torch.library.custom_op("mlx::int8_int4_affine_gemm", mutates_args=())
def int8_int4_affine_gemm(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, w_bias: Tensor, w_sum: Tensor
) -> Tensor:
    K = x.shape[-1]
    xq, x_scale, x_zero = quantize_per_token_int8_affine(x.reshape(-1, K))
    acc = xq.float() @ dequantize_int4_affine(w_packed, w_scale, w_bias).T
    y = x_scale * (acc - x_zero * w_sum.float())
    return y.to(x.dtype).reshape(*x.shape[:-1], w_packed.shape[0])


@torch.library.register_fake("mlx::int8_int4_affine_gemm")
def int8_int4_affine_gemm_fake(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, w_bias: Tensor, w_sum: Tensor
) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_packed.shape[0]))


@torch.library.custom_op("mlx::int8sym_int4_affine_gemm", mutates_args=())
def int8sym_int4_affine_gemm(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, w_bias: Tensor
) -> Tensor:
    K = x.shape[-1]
    xq, x_scale = quantize_per_token_int8(x.reshape(-1, K))
    y = x_scale * (xq.float() @ dequantize_int4_affine(w_packed, w_scale, w_bias).T)
    return y.to(x.dtype).reshape(*x.shape[:-1], w_packed.shape[0])


@torch.library.register_fake("mlx::int8sym_int4_affine_gemm")
def int8sym_int4_affine_gemm_fake(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, w_bias: Tensor
) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_packed.shape[0]))


@torch.library.custom_op("mlx::fp_int4_affine_gemm", mutates_args=())
def fp_int4_affine_gemm(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, w_bias: Tensor
) -> Tensor:
    y = x.float() @ dequantize_int4_affine(w_packed, w_scale, w_bias).T
    return y.to(x.dtype)


@torch.library.register_fake("mlx::fp_int4_affine_gemm")
def fp_int4_affine_gemm_fake(
    x: Tensor, w_packed: Tensor, w_scale: Tensor, w_bias: Tensor
) -> Tensor:
    return x.new_empty((*x.shape[:-1], w_packed.shape[0]))


# ---------------------------------------------------------------------------
# MLX handler
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_helpers import (
    emit_ceil_div,
    emit_floordiv,
    emit_if_else,
    emit_product,
    emit_shape,
    emit_sub_int,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
)

_K_ALIGN = 64
_PROLOGUE_THREADS = 256
_MATVEC_SIMDGROUPS = 2
_MATVEC_ROWS_PER_SIMDGROUP = 4
_MAX_M = 2**31 - 1
# Largest BM * BN that adds the bias term through a threadgroup tile (BIAS 1):
# the fp32 tile is 16 KB here; bigger tiles add it after the group loop.
_MAX_BIAS_TILE = 4096

# Distinct accumulator rows per thread of a single-SIMD-group int8 x int4b MPP
# op with an SM x SN tile (measured on M5 Max, EXPLORATION.md E7 / E12a). The
# kernel checks the layout at runtime and falls back to per-element code for
# tiles, dtypes or GPUs where it differs.
_SG_ROWS_PER_THREAD = {(32, 32): 4, (16, 64): 2, (16, 32): 2}

# Muse Glimmer (N, K) -> large-M threadgroup tile (BM, BN) of 32 x 32
# per-SIMD-group ops, the M5 Max int8 x int4 gs256 winners (EXPLORATION.md E23
# / E24). Other shapes and M tiers derive their tiles from the channelwise
# plans.
_MG_LARGE_M_TILES: dict[tuple[int, int], tuple[int, int]] = {
    (8704, 6656): (64, 128),  # qkv
    (6656, 4096): (32, 128),  # o
    (39936, 6656): (32, 256),  # gate_up
    (6656, 19968): (32, 128),  # down
    (202048, 6656): (32, 256),  # lm_head
}


class AffinePlan(NamedTuple):
    """Kernel choice for ``M <= max_m``. ``bn == 0`` selects the matvec kernel
    with ``bm`` activation rows per threadgroup; ``bn < 0`` the qmv_wide-style
    kernel with ``bm`` rows per threadgroup and ``-bn`` lanes per output
    channel; otherwise the MPP GEMM on a ``bm x bn`` threadgroup tile of
    ``sm x sn`` per-SIMD-group ops (``sm == 0``: one op shared by 4 SIMD
    groups, for 8-row tiles), split over K when ``splits > 1``. Grouped
    weights add the bias term with one MPP call (``bias == 1``) or after the
    group loop (``bias == 3``); per-channel weights fold it per element
    (``bias == 0``)."""

    max_m: int
    bm: int
    bn: int
    splits: int = 1
    sm: int = 32
    sn: int = 32
    bias: int = 0


def effective_splits(n_groups: int, splits: int) -> int:
    """Split count for grouped weights: each split takes ceil(G / splits)
    groups (fewer in the last), and splits that would be empty are dropped, as
    in the M5 Max study's kernel."""
    return math.ceil(n_groups / math.ceil(n_groups / splits))


def _mpp_plan(max_m: int, bm: int, bn: int, splits: int, N: int, K: int, gs: int):
    if max_m == _MAX_M and (N, K) in _MG_LARGE_M_TILES:
        bm, bn = _MG_LARGE_M_TILES[(N, K)]
    # 8-row tiles share one op across the threadgroup (the channelwise
    # kernels' layout); otherwise sub-tiles with measured accumulator layouts:
    # 16-row tiles as 16 x 64 / 16 x 32, larger ones as 32 x 32.
    if bm <= 8:
        while bn > 32 and bn > N:
            bn //= 2
        if gs < K:
            splits = effective_splits(K // gs, splits)
        return AffinePlan(max_m, bm, bn, splits, 0, 0, 3 if gs < K else 0)
    if bm <= 16:
        bm, sm, sn = 16, 16, 64 if bn >= 64 else 32
    else:
        sm = sn = 32
    while bn > sn and bn > N:
        bn //= 2
    bias = 0
    if gs < K:
        splits = effective_splits(K // gs, splits)
        bias = 1 if bm * bn <= _MAX_BIAS_TILE else 3
    return AffinePlan(max_m, bm, bn, splits, sm, sn, bias)


# Swept Muse Glimmer plans per (act, "pc" | "gs<GS>") and (N, K), as
# AffinePlan fields; regenerate with bench_muse_glimmer_gemm.py --emit-plans.
# Generated by bench_muse_glimmer_gemm.py --emit-plans from fp16 sweeps on Apple M4 Max.
_TUNED_PLANS: dict[tuple[str, str], dict[tuple[int, int], list[tuple]]] = {
    ("fp", "gs256"): {
        (6656, 4096): [  # o
            (1, 1, -8, 1),
            (2, 2, -32, 1),
            (4, 2, -8, 1),
            (16, 8, 32, 8, 0, 0, 3),
            (256, 32, 32, 1, 32, 32, 1),
            (_MAX_M, 64, 64, 1, 32, 32, 1),
        ],
        (6656, 19968): [  # down
            (1, 1, 0, 1),
            (2, 2, -16, 1),
            (4, 2, -8, 1),
            (16, 8, 32, 2, 0, 0, 3),
            (256, 128, 32, 2, 32, 32, 1),
            (_MAX_M, 32, 64, 1, 32, 32, 1),
        ],
        (8704, 6656): [  # qkv
            (1, 1, -8, 1),
            (2, 2, -32, 1),
            (4, 2, -16, 1),
            (16, 8, 64, 2, 0, 0, 3),
            (256, 32, 64, 2, 32, 32, 1),
            (_MAX_M, 32, 64, 1, 32, 32, 1),
        ],
        (39936, 6656): [  # gate_up
            (1, 1, -32, 1),
            (2, 2, -16, 1),
            (16, 8, 128, 2, 0, 0, 3),
            (256, 32, 64, 1, 32, 32, 1),
            (_MAX_M, 32, 32, 1, 32, 32, 1),
        ],
        (202048, 6656): [  # lm_head
            (1, 1, -32, 1),
            (2, 2, -16, 1),
            (16, 8, 128, 2, 0, 0, 3),
            (256, 64, 64, 1, 32, 32, 1),
            (_MAX_M, 128, 32, 1, 32, 32, 1),
        ],
    },
    ("fp", "pc"): {
        (6656, 4096): [  # o
            (1, 1, -16, 1),
            (2, 2, -16, 1),
            (4, 2, -8, 1),
            (16, 8, 32, 8, 0, 0, 0),
            (_MAX_M, 32, 32, 2, 32, 32, 0),
        ],
        (6656, 19968): [  # down
            (1, 1, -16, 1),
            (2, 2, -32, 1),
            (4, 2, 0, 1),
            (16, 8, 64, 8, 0, 0, 0),
            (_MAX_M, 128, 32, 4, 32, 32, 0),
        ],
        (8704, 6656): [  # qkv
            (1, 1, -8, 1),
            (2, 2, -16, 1),
            (4, 2, -8, 1),
            (16, 8, 64, 8, 0, 0, 0),
            (_MAX_M, 32, 64, 2, 32, 32, 0),
        ],
        (39936, 6656): [  # gate_up
            (1, 1, -8, 1),
            (4, 2, -16, 1),
            (16, 8, 128, 4, 0, 0, 0),
            (_MAX_M, 32, 64, 1, 32, 32, 0),
        ],
        (202048, 6656): [  # lm_head
            (1, 1, -16, 1),
            (4, 2, -16, 1),
            (16, 8, 128, 2, 0, 0, 0),
            (_MAX_M, 32, 128, 1, 32, 32, 0),
        ],
    },
    ("int8", "gs256"): {
        (6656, 4096): [  # o
            (2, 1, 0, 1),
            (4, 8, 128, 8, 0, 0, 3),
            (16, 8, 64, 8, 0, 0, 3),
            (_MAX_M, 64, 32, 1, 32, 32, 1),
        ],
        (6656, 19968): [  # down
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (4, 4, 0, 1),
            (16, 8, 32, 2, 0, 0, 3),
            (256, 128, 32, 2, 32, 32, 1),
            (_MAX_M, 64, 32, 1, 32, 32, 3),
        ],
        (8704, 6656): [  # qkv
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (4, 8, 128, 2, 0, 0, 3),
            (16, 8, 64, 2, 0, 0, 3),
            (256, 32, 64, 2, 32, 32, 1),
            (_MAX_M, 32, 32, 1, 32, 32, 1),
        ],
        (39936, 6656): [  # gate_up
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (16, 8, 128, 2, 0, 0, 3),
            (256, 128, 32, 1, 32, 32, 1),
            (_MAX_M, 64, 32, 1, 32, 32, 1),
        ],
        (202048, 6656): [  # lm_head
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (16, 8, 128, 2, 0, 0, 3),
            (256, 128, 32, 1, 32, 32, 1),
            (_MAX_M, 64, 32, 1, 32, 32, 1),
        ],
    },
    ("int8", "pc"): {
        (6656, 4096): [  # o
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (16, 8, 64, 8, 0, 0, 0),
            (_MAX_M, 32, 64, 2, 32, 32, 0),
        ],
        (6656, 19968): [  # down
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (4, 8, 64, 8, 0, 0, 0),
            (16, 8, 128, 4, 0, 0, 0),
            (_MAX_M, 32, 32, 4, 32, 32, 0),
        ],
        (8704, 6656): [  # qkv
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (4, 8, 128, 8, 0, 0, 0),
            (16, 8, 32, 2, 0, 0, 0),
            (_MAX_M, 32, 64, 4, 32, 32, 0),
        ],
        (39936, 6656): [  # gate_up
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (4, 8, 128, 8, 0, 0, 0),
            (16, 8, 128, 2, 0, 0, 0),
            (_MAX_M, 32, 64, 1, 32, 32, 0),
        ],
        (202048, 6656): [  # lm_head
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (16, 8, 128, 2, 0, 0, 0),
            (_MAX_M, 32, 64, 1, 32, 32, 0),
        ],
    },
}


def affine_plans(N: int, K: int, group_size: int, act: str) -> list[AffinePlan]:
    """Kernel plans, smallest ``max_m`` first; the last covers every larger M.
    Swept shapes and group sizes use ``_TUNED_PLANS``. Others start from the
    channelwise ``fp_int4_gemm`` / ``int8_intx_gemm`` plans;
    MPP tiles become per-SIMD-group tiles (Muse Glimmer's large-M tiles from
    the M5 Max group-wise study) with splits made effective (no empty split).
    Small-M int8 is weight-bandwidth bound like fp, so int8 matvec plans take
    fp_int4's swept wide plan for the same ``max_m`` where there is one.
    ``int8sym`` runs the int8 GEMM kernels and uses int8's plans unless it has
    its own."""
    scheme = "pc" if group_size == K else f"gs{group_size}"
    tuned = _TUNED_PLANS.get((act, scheme), {}).get((N, K))
    if tuned is None and act == "int8sym":
        tuned = _TUNED_PLANS.get(("int8", scheme), {}).get((N, K))
    if tuned is not None:
        return [AffinePlan(*p) for p in tuned]
    if act != "fp":
        fp_wide = {p.max_m: p for p in fp_int4_gemm.full_k_plans(N, K) if p.bn < 0}
        base = [
            fp_wide.get(p.max_m, p) if p.bn == 0 else p
            for p in int8_intx_gemm.full_k_plans(N, K)
        ]
    else:
        base = fp_int4_gemm.full_k_plans(N, K)
    return [
        (
            AffinePlan(p.max_m, p.bm, p.bn)
            if p.bn <= 0
            else _mpp_plan(p.max_m, p.bm, p.bn, p.splits, N, K, group_size)
        )
        for p in base
    ]


_HEADER = f"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant float INT4_AFFINE_EPS = {_EPS}f;

// Applies a weight scale and bias to a dot product d with activation sum sx.
// Every per-channel kernel goes through this one expression, so per-channel
// int8 results, whose d and sx are exact integers, are bit-identical across
// kernels.
inline float fold_qparams(float ws, float wb, float d, float sx) {{
  return fma(wb, sx, ws * d);
}}
"""

# Per activation mode substitutions into the kernel templates below. In the
# epilogues, ``v`` is the weight-folded accumulator for output (m, n) and the
# vector forms take ``float4 v`` for columns n .. n + 3; ``x_load`` reads 16
# activations at ``xrow + k`` for ``x_vec`` to convert. ``exact_pc`` keeps
# per-channel sums exact (int8) instead of folding the bias into the decode.
_ACT_SUBS = {
    "int8": {
        "x_elem": "int8_t",
        "x_ptr": "xq",
        "acc_t": "int",
        "x_rows": "xq_shape[0]",
        "def_m": "const int M = x_scale_shape[0];",
        "x_load": "const uint4 xv = *(device const uint4*)(xrow + k);",
        "x_vec": "float4(as_type<char4>(xv[i]))",
        "exact_pc": "true",
        "out_val": "T(x_scale[m] * fma(-x_zero[m], float(w_sum[n]), v))",
        "out_vec": (
            "vec<T, 4>(x_scale[m] * fma(float4(-x_zero[m]), float4(float(w_sum[n]), "
            "float(w_sum[n + 1]), float(w_sum[n + 2]), float(w_sum[n + 3])), v))"
        ),
    },
    "int8sym": {
        "x_elem": "int8_t",
        "x_ptr": "xq",
        "acc_t": "int",
        "x_rows": "xq_shape[0]",
        "def_m": "const int M = x_scale_shape[0];",
        "x_load": "const uint4 xv = *(device const uint4*)(xrow + k);",
        "x_vec": "float4(as_type<char4>(xv[i]))",
        "exact_pc": "true",
        "out_val": "T(x_scale[m] * v)",
        "out_vec": "vec<T, 4>(x_scale[m] * v)",
    },
    "fp": {
        "x_elem": "T",
        "x_ptr": "x",
        "acc_t": "float",
        "x_rows": "M",
        "def_m": """int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }""",
        "x_load": "device const vec<T, 4>* xv = (device const vec<T, 4>*)(xrow + k);",
        "x_vec": "float4(xv[i])",
        "exact_pc": "false",
        "out_val": "T(v)",
        "out_vec": "vec<T, 4>(v)",
    },
}


# Template args T, K, TPR, XG: one threadgroup of TPR threads per xq row.
# Writes xq, x_scale (and x_zero for affine activations), and x_sum[m, b] = sum
# of xq over the b-th XG-wide K block (XG % 64 == 0) as fp32 (exact). Padding
# rows past M write zeros to xq and x_sum so the GEMM sees whole BM tiles.
# Symmetric activations use x_scale = max(max |x|, eps) / 127 and
# xq = clamp(round(x / x_scale), -128, 127), as int8_intx_gemm.
_INT8_QUANT_TEMPLATE = Template(
    """
    constexpr int SG = TPR / 32;
    constexpr int NXG = K / XG;
    threadgroup float sg_lo[SG];
    threadgroup float sg_hi[SG];
    threadgroup atomic_int sums[NXG];

    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int m = int(threadgroup_position_in_grid.y);
    const int tid = int(thread_index_in_threadgroup);
    device uint* xq_row = (device uint*)(xq + size_t(m) * K);
    if (m >= M) {
        for (int i = tid; i < K / 4; i += TPR) {
            xq_row[i] = 0;
        }
        for (int b = tid; b < NXG; b += TPR) {
            x_sum[size_t(m) * NXG + b] = 0.0f;
        }
        return;
    }
    device const vec<T, 4>* x_row = (device const vec<T, 4>*)(x + size_t(m) * K);
    for (int b = tid; b < NXG; b += TPR) {
        atomic_store_explicit(&sums[b], 0, memory_order_relaxed);
    }

    float lo = 0.0f;
    float hi = 0.0f;
    for (int i = tid; i < K / 4; i += TPR) {
        const float4 v = float4(x_row[i]);
        lo = min(lo, min(min(v.x, v.y), min(v.z, v.w)));
        hi = max(hi, max(max(v.x, v.y), max(v.z, v.w)));
    }
    lo = simd_min(lo);
    hi = simd_max(hi);
    if (thread_index_in_simdgroup == 0) {
        sg_lo[simdgroup_index_in_threadgroup] = lo;
        sg_hi[simdgroup_index_in_threadgroup] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lo = sg_lo[0];
    hi = sg_hi[0];
    for (int s = 1; s < SG; ++s) {
        lo = min(lo, sg_lo[s]);
        hi = max(hi, sg_hi[s]);
    }
    const float scale = $scale;
    const float inv_scale = precise::divide(1.0f, scale);

    // Each 16-lane half of a SIMD group covers one 64-value segment, which
    // lies within a single XG block; the loop bound is uniform so the
    // shuffles see every lane.
    for (int base = 0; base < K / 4; base += TPR) {
        const int i = base + tid;
        int sum = 0;
        if (i < K / 4) {
            const float4 q = $quant;
            xq_row[i] = as_type<uint>(char4(q));
            sum = int((q.x + q.y) + (q.z + q.w));
        }
        for (ushort off = 8; off > 0; off /= 2) {
            sum += simd_shuffle_down(sum, off);
        }
        if (i < K / 4 && thread_index_in_simdgroup % 16 == 0) {
            atomic_fetch_add_explicit(&sums[4 * i / XG], sum, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = tid; b < NXG; b += TPR) {
        x_sum[size_t(m) * NXG + b] = float(atomic_load_explicit(&sums[b], memory_order_relaxed));
    }
    if (tid == 0) {
        x_scale[m] = scale;$zero
    }
"""
)
_INT8_QUANT_SOURCE = _INT8_QUANT_TEMPLATE.substitute(
    scale="max(precise::divide(hi - lo, 255.0f), INT4_AFFINE_EPS)",
    quant="clamp(rint((float4(x_row[i]) - lo) * inv_scale), 0.0f, 255.0f) - 128.0f",
    zero="\n        x_zero[m] = -128.0f - precise::divide(lo, scale);",
)
_INT8_SYM_QUANT_SOURCE = _INT8_QUANT_TEMPLATE.substitute(
    scale="max(precise::divide(max(hi, -lo), 127.0f), INT4_AFFINE_EPS)",
    quant="clamp(rint(float4(x_row[i]) * inv_scale), -128.0f, 127.0f)",
    zero="",
)


# Template args T, K, TPR, XG: one SIMD group per (row, XG-wide K block) writes
# x_sum[m, b] = sum of x over the block, or 0 for padding rows past M.
_FP_SUM_SOURCE = """
    constexpr int NXG = K / XG;
    int M = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
        M *= x_shape[d];
    }
    const int m = int(threadgroup_position_in_grid.y);
    const int b = int(threadgroup_position_in_grid.x) * (TPR / 32)
        + int(simdgroup_index_in_threadgroup);
    if (b >= NXG) {
        return;
    }
    if (m >= M) {
        if (thread_index_in_simdgroup == 0) {
            x_sum[size_t(m) * NXG + b] = 0.0f;
        }
        return;
    }
    device const vec<T, 4>* xb = (device const vec<T, 4>*)(x + size_t(m) * K + size_t(b) * XG);
    float sum = 0.0f;
    for (int i = int(thread_index_in_simdgroup); i < XG / 4; i += 32) {
        const float4 v = float4(xb[i]);
        sum += (v.x + v.y) + (v.z + v.w);
    }
    sum = simd_sum(sum);
    if (thread_index_in_simdgroup == 0) {
        x_sum[size_t(m) * NXG + b] = sum;
    }
"""


# Template args T, ST, N, K, BM, BN, SM, SN, NM, GS, SPLITS, BIAS, COOP. Each
# SIMD group of a BM x BN threadgroup runs its own SM x SN MPP op
# (execution_simdgroup, so SIMD groups do not rescale in lockstep), or with COOP
# the threadgroup's 4 SIMD groups share one BM x BN op (execution_simdgroups<4>,
# as in int8_intx_gemm / fp_int4_gemm), which allows 8-row tiles; COOP always
# runs the per-element path below. The last column tile shifts back
# to N - BN so full tiles stay in bounds (overlapping columns are written twice,
# identically). K is walked in KB-wide blocks matching x_sum's: weight groups
# (KB = GS), or one K / SPLITS block per split for per-channel weights. x_sum
# has the padded tile rows (at least 16), so MLX never passes it in constant
# memory (inputs under 8 elements), where it could not back an MPP tensor; M
# comes from x_scale (int8) or x (fp).
#
# Grouped (BIAS 1 / 3): grid z picks ceil(K / GS / SPLITS) consecutive groups
# (fewer in the last split; effective_splits keeps every split non-empty); per
# group MPP multiplies the [KB, rows] activation block by the packed [KB, N]
# weight block (W transposed) into ``acc`` and facc = fma(acc, w_scale, facc).
# The bias term sum_g S[m, g] * w_bias[g, n] is one f32 x ST MPP call per
# sub-tile handed to ``facc`` through threadgroup memory (BIAS 1), or one pass
# over the groups after the loop (BIAS 3). The split's unscaled sums go to
# ``partial`` for the reduce kernel when SPLITS > 1.
#
# Per-channel (BIAS 0): one MPP call over the split's K block. Unsplit, the
# channel's scale and bias fold per element through fold_qparams; split (RAW),
# the unscaled ``acc`` (INT32 for int8) goes to ``partial`` and the reduce
# kernel folds once.
#
# Fast path: the destination layout of a single-SIMD-group op (measured for
# int8 x int4b at 32 x 32, 16 x 64, 16 x 32) puts element i at column
# rn[(i / 8) % NRN] + i % 4 and row rm[(i / (8 * NRN)) * 2 + (i / 4) % 2], with
# NM rows per thread and NRN = CAP / (4 * NM) runs of 4 adjacent columns, so a
# thread loads each group's scales as NRN 4-vectors and stores 4-vectors. Each
# SIMD group checks this layout once (plus N % 4 == 0 and a full column
# sub-tile) and otherwise runs the same math per element. The layout was
# measured on M5 Max; on M4 Max it fails for all three sub-tiles, so the
# per-element path is the one that runs there.
#
# Operands (operand_setup / operand_block from _operand_views): each K
# block is a fixed-extent slice of the full tensor where every tile is known to
# be in bounds, as the M5 Max study's kernel does (int8 activations, whose rows
# are padded to whole tiles; weights when N >= BN, where the column-tile shift
# keeps every sub-tile inside N), so MPP skips bounds handling; otherwise a
# bounds-checked view of exactly KB columns (fp activations; weights when
# N < BN, the ``_dynw`` kernels).
_MPP_TEMPLATE = Template(
    """
    constexpr bool PER_CHANNEL = GS == K;
    constexpr int KB = PER_CHANNEL ? K / SPLITS : GS;
    constexpr int NB = K / KB;
    constexpr int NBS = (NB + SPLITS - 1) / SPLITS;
    constexpr bool RAW = PER_CHANNEL && SPLITS > 1;
    static_assert(!(COOP && BIAS == 1), "cooperative tiles add the bias after the loop");
    constexpr int SGN = COOP ? 1 : BN / SN;
    constexpr int CAP = SM * SN / 32;
    constexpr int NRN = CAP / (4 * NM) > 0 ? CAP / (4 * NM) : 1;
    $def_m
    const int rows = $x_rows;
    const int s = int(threadgroup_position_in_grid.z);
    const int b0 = s * NBS;
    const int nbs = min(NBS, NB - b0);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int tn = int(threadgroup_position_in_grid.x) * BN;
    const int n0 = (N >= BN ? min(tn, N - BN) : tn) + (COOP ? 0 : (sg % SGN) * SN);
    const int m0 = int(threadgroup_position_in_grid.y) * BM + (COOP ? 0 : (sg / SGN) * SM);
    if (m0 >= M || n0 >= N) {
        return;
    }

    using XTensor = tensor<device $x_elem, dextents<int, 2>, tensor_inline>;
    using WTensor = tensor<device int4b_format, dextents<int, 2>, tensor_inline>;
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, KB, false, true, false, matmul2d_descriptor::mode::multiply);
    using Scope = metal::conditional_t<COOP, execution_simdgroups<4>, execution_simdgroup>;
    matmul2d<desc, Scope> mm;
$operand_setup
    auto acc = mm.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), $acc_t>();
    auto facc = mm.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < facc.get_capacity(); ++i) {
        facc[i] = 0.0f;
    }

    int rn[NRN];
    int rm[NM];
    bool fast = !COOP && N % 4 == 0 && n0 + SN <= N && facc.get_capacity() == CAP;
    if (fast) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < NRN; ++r) {
            rn[r] = n0 + int(facc.get_multidimensional_index(r * 8)[0]);
        }
        #pragma clang loop unroll(full)
        for (int t = 0; t < NM; ++t) {
            rm[t] = m0 + int(facc.get_multidimensional_index((t / 2) * NRN * 8 + (t % 2) * 4)[1]);
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < CAP; ++i) {
            const auto c = facc.get_multidimensional_index(i);
            const int r = (i / 8) % NRN;
            const int t = (i / (8 * NRN)) * 2 + (i / 4) % 2;
            fast = fast && facc.is_valid_element(i) && n0 + int(c[0]) == rn[r] + i % 4
                && m0 + int(c[1]) == rm[t];
        }
    }
    // Every lane must take the same path around the SIMD group's MPP calls.
    fast = simd_all(fast);

    if constexpr (BIAS == 1) {
        // Cooperative tensors of different ops cannot be mixed, so the
        // f32 x ST result reaches facc through this SIMD group's slice of a
        // fp32 tile.
        threadgroup float bias_tile[BM * BN];
        tensor<device float, dextents<int, 2>, tensor_inline> XS(
            (device float*)x_sum + b0, dextents<int, 2>(nbs, x_sum_shape[0]),
            array<int, 2>{1, NB});
        tensor<device ST, dextents<int, 2>, tensor_inline> WB(
            (device ST*)w_bias + size_t(b0) * N, dextents<int, 2>(N, nbs), array<int, 2>{1, N});
        constexpr auto bdesc = matmul2d_descriptor(
            SM, SN, static_cast<int>(dynamic_extent), false, false, false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<bdesc, execution_simdgroup> mb;
        auto xa = XS.slice(0, m0);
        auto wbt = WB.slice(n0, 0);
        auto bacc = mb.template get_destination_cooperative_tensor<decltype(xa), decltype(wbt), float>();
        mb.run(xa, wbt, bacc);
        tensor<threadgroup float, extents<int, SN, SM>, tensor_inline> bt(
            bias_tile + sg * SM * SN, extents<int, SN, SM>{});
        bacc.store(bt);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        facc.load(bt);
    }

    // As in the study's kernel, the fast and per-element paths run separate
    // group loops, so the hot loop holds only the MPP call and the rescale.
    if (fast) {
        vec<ST, 4> sv[NRN];
        if (!PER_CHANNEL) {
            #pragma clang loop unroll(full)
            for (int r = 0; r < NRN; ++r) {
                sv[r] = *(device const vec<ST, 4>*)(w_scale + size_t(b0) * N + rn[r]);
            }
        }
        for (int b = b0; b < b0 + nbs; ++b) {
            // The group's scales, with the next group's loaded behind them.
            vec<ST, 4> sc[NRN];
            if (!PER_CHANNEL) {
                const int bnext = min(b + 1, NB - 1);
                #pragma clang loop unroll(full)
                for (int r = 0; r < NRN; ++r) {
                    sc[r] = sv[r];
                    sv[r] = *(device const vec<ST, 4>*)(w_scale + size_t(bnext) * N + rn[r]);
                }
            }
$operand_block
            mm.run(mA, mB, acc);
            if (PER_CHANNEL) {
                break;
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < CAP; ++i) {
                facc[i] = fma(float(acc[i]), float(sc[(i / 8) % NRN][i % 4]), facc[i]);
            }
        }
    } else {
        for (int b = b0; b < b0 + nbs; ++b) {
$operand_block
            mm.run(mA, mB, acc);
            if (PER_CHANNEL) {
                break;
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) {
                    const auto c = acc.get_multidimensional_index(i);
                    const int n = n0 + int(c[0]);
                    const int m = m0 + int(c[1]);
                    // Rows past M read x_sum's zero padding and are never
                    // stored; columns stay below N unless N < BN.
                    if (N >= BN || n < N) {
                        const size_t wi = size_t(b) * N + n;
                        facc[i] = fma(float(acc[i]), float(w_scale[wi]), facc[i]);
                        if (BIAS == 3) {
                            facc[i] = fma(x_sum[size_t(m) * NB + b], float(w_bias[wi]), facc[i]);
                        }
                    }
                }
            }
        }
    }

    if (BIAS == 3 && fast) {
        for (int b = b0; b < b0 + nbs; ++b) {
            vec<ST, 4> bv[NRN];
            float xv[NM];
            #pragma clang loop unroll(full)
            for (int r = 0; r < NRN; ++r) {
                bv[r] = *(device const vec<ST, 4>*)(w_bias + size_t(b) * N + rn[r]);
            }
            #pragma clang loop unroll(full)
            for (int t = 0; t < NM; ++t) {
                xv[t] = rm[t] < M ? x_sum[size_t(rm[t]) * NB + b] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < CAP; ++i) {
                const int r = (i / 8) % NRN;
                const int t = (i / (8 * NRN)) * 2 + (i / 4) % 2;
                facc[i] = fma(xv[t], float(bv[r][i % 4]), facc[i]);
            }
        }
    }

    if (PER_CHANNEL && !RAW) {
        if (fast) {
            vec<ST, 4> ws4[NRN];
            vec<ST, 4> wb4[NRN];
            float xv[NM];
            #pragma clang loop unroll(full)
            for (int r = 0; r < NRN; ++r) {
                ws4[r] = *(device const vec<ST, 4>*)(w_scale + rn[r]);
                wb4[r] = *(device const vec<ST, 4>*)(w_bias + rn[r]);
            }
            #pragma clang loop unroll(full)
            for (int t = 0; t < NM; ++t) {
                xv[t] = rm[t] < M ? x_sum[rm[t]] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < CAP; ++i) {
                const int r = (i / 8) % NRN;
                const int t = (i / (8 * NRN)) * 2 + (i / 4) % 2;
                facc[i] = fold_qparams(
                    float(ws4[r][i % 4]), float(wb4[r][i % 4]), float(acc[i]), xv[t]);
            }
        } else {
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) {
                    const auto c = acc.get_multidimensional_index(i);
                    const int n = n0 + int(c[0]);
                    const int m = m0 + int(c[1]);
                    if (m < M && n < N) {
                        facc[i] = fold_qparams(
                            float(w_scale[n]), float(w_bias[n]), float(acc[i]), x_sum[m]);
                    }
                }
            }
        }
    }

    if (fast) {
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < CAP / 4; ++j) {
            const int r = (j / 2) % NRN;
            const int t = (j / (2 * NRN)) * 2 + j % 2;
            const int m = rm[t];
            const int n = rn[r];
            if (m < M) {
                $store_fast;
            }
        }
    } else {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                const auto c = acc.get_multidimensional_index(i);
                const int n = n0 + int(c[0]);
                const int m = m0 + int(c[1]);
                if (m < M && n < N) {
                    const float v = facc[i];
                    $store_slow;
                }
            }
        }
    }
"""
)

_MPP_STORES = {
    "full_k": {
        "store_fast": """const float4 v = float4(
                    facc[4 * j], facc[4 * j + 1], facc[4 * j + 2], facc[4 * j + 3]);
                *(device vec<T, 4>*)(out + size_t(m) * N + n) = $out_vec""",
        "store_slow": "out[size_t(m) * N + n] = $out_val",
    },
    "split_k": {
        "store_fast": """const size_t pi = (size_t(s) * M + m) * N + n;
                if (RAW) {
                    #pragma clang loop unroll(full)
                    for (int q = 0; q < 4; ++q) {
                        partial[pi + q] = acc[4 * j + q];
                    }
                } else {
                    *(device float4*)(partial + pi) = float4(
                        facc[4 * j], facc[4 * j + 1], facc[4 * j + 2], facc[4 * j + 3]);
                }""",
        "store_slow": """const size_t pi = (size_t(s) * M + m) * N + n;
                    if (RAW) {
                        partial[pi] = acc[i];
                    } else {
                        partial[pi] = v;
                    }""",
    },
}

# Template args T, N, SPLITS: one thread per VW adjacent outputs (VW = 4 when
# N % 4 == 0) sums the grouped kernel's FP32 partials.
_REDUCE_TEMPLATE = Template(
    """
    constexpr int VW = N % 4 == 0 ? 4 : 1;
    const int M = partial_shape[1];
    const size_t q = thread_position_in_grid.x;
    if (q >= size_t(M) * (N / VW)) {
        return;
    }
    const int m = int(q / (N / VW));
    const int nv = int(q % (N / VW)) * VW;
    #pragma clang loop unroll(full)
    for (int c = 0; c < VW; ++c) {
        const int n = nv + c;
        float v = 0.0f;
        for (int s = 0; s < SPLITS; ++s) {
            v += partial[(size_t(s) * M + m) * N + n];
        }
        out[size_t(m) * N + n] = $out_val;
    }
"""
)

# Template args T, N, SPLITS for per-channel split-K: one thread per VW
# adjacent outputs sums the unscaled partials (INT32 for int8) and the splits'
# x_sum blocks, then folds the channel's scale and bias once.
_REDUCE_RAW_TEMPLATE = Template(
    """
    constexpr int VW = N % 4 == 0 ? 4 : 1;
    const int M = partial_shape[1];
    const size_t q = thread_position_in_grid.x;
    if (q >= size_t(M) * (N / VW)) {
        return;
    }
    const int m = int(q / (N / VW));
    const int nv = int(q % (N / VW)) * VW;
    $acc_t sx = 0;
    for (int s = 0; s < SPLITS; ++s) {
        sx += $acc_t(x_sum[size_t(m) * SPLITS + s]);
    }
    #pragma clang loop unroll(full)
    for (int c = 0; c < VW; ++c) {
        const int n = nv + c;
        $acc_t d = 0;
        for (int s = 0; s < SPLITS; ++s) {
            d += partial[(size_t(s) * M + m) * N + n];
        }
        const float v = fold_qparams(float(w_scale[n]), float(w_bias[n]), float(d), float(sx));
        out[size_t(m) * N + n] = $out_val;
    }
"""
)


# Matvec (template args T, N, K, MR, GS; K % 512 == 0) for small M, laid out
# like int8_intx_gemm's: each SIMD group computes 4 output channels for up to
# MR activation rows; each lane covers 16 K values per 512-wide step, all in
# one weight group. XOR 0x88 makes nibbles unsigned (q = w + 8) so masked
# ushort words convert directly to float against activations pre-scaled by
# 1, 1/16, 1/256, 1/4096. Grouped weights: each lane folds its 16-value partial
# dot product with its group's scale and (bias - 8 * scale) times its
# activation sum. Per-channel: lanes accumulate the unscaled dot product and
# activation sum (exact integers below 2^24 for int8), reduce them (INT32 for
# int8, as in int8_intx_gemm), and fold once.
_MATVEC_TEMPLATE = Template(
    """
    constexpr int SG = 2;
    constexpr int RPS = 4;
    constexpr int VPT = 16;
    constexpr int BLOCK = 32 * VPT;
    constexpr int KP = K / 2;
    constexpr bool PER_CHANNEL = GS == K;

    $def_m
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int n0 = (int(threadgroup_position_in_grid.x) * SG + sg) * RPS;
    const int m0 = int(threadgroup_position_in_grid.y) * MR;
    if (n0 >= N) {
        return;
    }
    const int mrows = min(MR, M - m0);

    device const uchar* w_rows[RPS];
    int nr[RPS];
    #pragma clang loop unroll(full)
    for (int r = 0; r < RPS; ++r) {
        nr[r] = min(n0 + r, N - 1);
        w_rows[r] = w_packed + size_t(nr[r]) * KP + lane * (VPT / 2);
    }
    device const $x_elem* x_base = (device const $x_elem*)$x_ptr + size_t(m0) * K + lane * VPT;
    const float4 prescale = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc[MR][RPS];
    float xacc[MR];
    #pragma clang loop unroll(full)
    for (int mi = 0; mi < MR; ++mi) {
        xacc[mi] = 0.0f;
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPS; ++r) {
            acc[mi][r] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += BLOCK) {
        const size_t gN = size_t((k + lane * VPT) / GS) * N;
        // Converted once per block and reused for every activation row.
        float4 wf[RPS][4];
        float ws[RPS];
        float wo[RPS];
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPS; ++r) {
            const ushort4 q = as_type<ushort4>(
                *(device const uint2*)(w_rows[r] + k / 2) ^ uint2(0x88888888u));
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                wf[r][i] = float4(
                    q[i] & 0x000f, q[i] & 0x00f0, q[i] & 0x0f00, q[i] & 0xf000);
            }
            if (!PER_CHANNEL) {
                ws[r] = float(w_scale[gN + nr[r]]);
                wo[r] = fma(-8.0f, ws[r], float(w_bias[gN + nr[r]]));
            }
        }
        #pragma clang loop unroll(full)
        for (int mi = 0; mi < MR; ++mi) {
            if (mi < mrows) {
                device const $x_elem* xrow = x_base + size_t(mi) * K;
                $x_load
                float4 xs[4];
                float xsum = 0.0f;
                #pragma clang loop unroll(full)
                for (int i = 0; i < 4; ++i) {
                    const float4 v = $x_vec;
                    xsum += (v.x + v.y) + (v.z + v.w);
                    xs[i] = v * prescale;
                }
                xacc[mi] += xsum;
                #pragma clang loop unroll(full)
                for (int r = 0; r < RPS; ++r) {
                    float a = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < 4; ++i) {
                        a += dot(xs[i], wf[r][i]);
                    }
                    acc[mi][r] += PER_CHANNEL ? a : fma(ws[r], a, wo[r] * xsum);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int mi = 0; mi < MR; ++mi) {
        if (mi < mrows) {
            const int m = m0 + mi;
            const $acc_t sx = PER_CHANNEL ? simd_sum($acc_t(xacc[mi])) : 0;
            #pragma clang loop unroll(full)
            for (int r = 0; r < RPS; ++r) {
                float v;
                if (PER_CHANNEL) {
                    const $acc_t d = simd_sum($acc_t(acc[mi][r] - 8.0f * xacc[mi]));
                    v = fold_qparams(
                        float(w_scale[nr[r]]), float(w_bias[nr[r]]), float(d), float(sx));
                } else {
                    v = simd_sum(acc[mi][r]);
                }
                const int n = n0 + r;
                if (lane == 0 && n < N) {
                    out[size_t(m) * N + n] = $out_val;
                }
            }
        }
    }
"""
)


# Template args T, N, K, NV, KL, GS (K % (16 * KL) == 0) for small M, laid out
# like fp_int4_gemm's qmv_wide-style kernel: KL lanes split K for one output
# channel (32 / KL channels per SIMD group, 2 SIMD groups per threadgroup) and
# each threadgroup streams NV activation rows, so each lane decodes one
# 16-value chunk (within one weight group) of one channel per step and reuses
# it for every row. The XOR / mask decode is undone on the weight side with
# one fma per weight, which for grouped weights also applies the group's scale
# and bias. Per-channel fp loads its channel's scale and bias once and applies
# them in that same fma, so it does no more work per step than fp_int4's
# kernel; per-channel int8 (EXACT_PC) decodes to the exact signed int4 values
# and also sums activations, so the unscaled dot product and activation sum
# reduce exactly in INT32 and fold once.
_WIDE_TEMPLATE = Template(
    """
    constexpr int SG = 2;
    constexpr int CPS = 32 / KL;
    constexpr int VPL = 16;
    constexpr int STEP = KL * VPL;
    constexpr int KP = K / 2;
    constexpr bool PER_CHANNEL = GS == K;
    constexpr bool EXACT_PC = PER_CHANNEL && $exact_pc;

    $def_m
    const int lane = int(thread_index_in_simdgroup);
    const int kl = lane % KL;
    const int n = (int(threadgroup_position_in_grid.x) * SG
                   + int(simdgroup_index_in_threadgroup)) * CPS + lane / KL;
    const int nc = min(n, N - 1);
    const int v0 = int(threadgroup_position_in_grid.y) * NV;

    device const uchar* w = w_packed + size_t(nc) * KP + kl * (VPL / 2);
    device const $x_elem* x_rows[NV];
    #pragma clang loop unroll(full)
    for (int j = 0; j < NV; ++j) {
        x_rows[j] = (device const $x_elem*)$x_ptr + size_t(min(v0 + j, M - 1)) * K + kl * VPL;
    }
    const float4 pre = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);

    float acc[NV];
    float xacc[NV];
    #pragma clang loop unroll(full)
    for (int j = 0; j < NV; ++j) {
        acc[j] = 0.0f;
        xacc[j] = 0.0f;
    }
    float4 pc_scale = pre;
    float pc_bias = -8.0f;
    if (PER_CHANNEL && !EXACT_PC) {
        const float ws = float(w_scale[nc]);
        pc_scale = pre * ws;
        pc_bias = fma(-8.0f, ws, float(w_bias[nc]));
    }

    for (int k = 0; k < K; k += STEP) {
        float4 wscale = pc_scale;
        float4 wbias = float4(pc_bias);
        if (!PER_CHANNEL) {
            const size_t wi = size_t((k + kl * VPL) / GS) * N + nc;
            const float ws = float(w_scale[wi]);
            wscale = pre * ws;
            wbias = float4(fma(-8.0f, ws, float(w_bias[wi])));
        }
        const ushort4 q = as_type<ushort4>(
            *(device const uint2*)(w + k / 2) ^ uint2(0x88888888u));
        float4 wf[4];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
            wf[i] = fma(
                float4(q[i] & 0x000f, q[i] & 0x00f0, q[i] & 0x0f00, q[i] & 0xf000),
                wscale,
                wbias);
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < NV; ++j) {
            device const $x_elem* xrow = x_rows[j];
            $x_load
            float4 xf[4];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                xf[i] = $x_vec;
                if (EXACT_PC) {
                    xacc[j] += (xf[i].x + xf[i].y) + (xf[i].z + xf[i].w);
                }
            }
            acc[j] += (dot(wf[0], xf[0]) + dot(wf[1], xf[1]))
                + (dot(wf[2], xf[2]) + dot(wf[3], xf[3]));
        }
    }

    // Reduce within this channel's KL lanes; simd_sum would mix channels.
    #pragma clang loop unroll(full)
    for (int j = 0; j < NV; ++j) {
        float vg = acc[j];
        $acc_t d = $acc_t(acc[j]);
        $acc_t sx = $acc_t(xacc[j]);
        #pragma clang loop unroll(full)
        for (ushort off = KL / 2; off > 0; off /= 2) {
            if (EXACT_PC) {
                d += simd_shuffle_down(d, off);
                sx += simd_shuffle_down(sx, off);
            } else {
                vg += simd_shuffle_down(vg, off);
            }
        }
        const int m = v0 + j;
        if (kl == 0 && n < N && m < M) {
            const float v = EXACT_PC
                ? fold_qparams(float(w_scale[n]), float(w_bias[n]), float(d), float(sx))
                : vg;
            out[size_t(m) * N + n] = $out_val;
        }
    }
"""
)


def _operand_views(static_x: bool, static_w: bool) -> dict[str, str]:
    """``operand_setup`` (the full tensors and block 0's views, which fix the
    accumulators' types) and ``operand_block`` (block b's views mA / mB) for
    the MPP template."""
    x_full = (
        "XTensor X((device $x_elem*)$x_ptr, dextents<int, 2>(K, rows), "
        "array<int, 2>{1, K});"
    )
    w_full = (
        "WTensor W((device uchar*)w_packed, dextents<int, 2>(K, N), "
        "array<int, 2>{1, K});"
    )
    if static_x:
        x_setup = x_full + "\n    auto mA0 = X.template slice<KB, SM>(0, m0);"
        x_block = "auto mA = X.template slice<KB, SM>(b * KB, m0);"
    else:
        x_setup = (
            "XTensor X0((device $x_elem*)$x_ptr, dextents<int, 2>(KB, rows), "
            "array<int, 2>{1, K});\n    auto mA0 = X0.slice(0, m0);"
        )
        x_block = (
            "XTensor Xb((device $x_elem*)$x_ptr + b * KB, dextents<int, 2>(KB, rows), "
            "array<int, 2>{1, K});\n        auto mA = Xb.slice(0, m0);"
        )
    if static_w:
        w_setup = w_full + "\n    auto mB0 = W.template slice<KB, SN>(0, n0);"
        w_block = "auto mB = W.template slice<KB, SN>(b * KB, n0);"
    else:
        w_setup = (
            "WTensor W0((device uchar*)w_packed, dextents<int, 2>(KB, N), "
            "array<int, 2>{1, K});\n    auto mB0 = W0.slice(0, n0);"
        )
        w_block = (
            "WTensor Wb((device uchar*)w_packed + b * (KB / 2), dextents<int, 2>(KB, N), "
            "array<int, 2>{1, K});\n        auto mB = Wb.slice(0, n0);"
        )
    return {
        "operand_setup": "    " + x_setup + "\n    " + w_setup,
        "operand_block": "            " + x_block + "\n            " + w_block,
    }


def _sources(act: str) -> dict[str, str]:
    subs = _ACT_SUBS[act]
    mpp = {}
    for kind, stores in _MPP_STORES.items():
        for suffix, static_w in (("", True), ("_dynw", False)):
            pieces = {**stores, **_operand_views(act != "fp", static_w)}
            mpp[kind + suffix] = _MPP_TEMPLATE.substitute(
                subs, **{k: Template(v).substitute(subs) for k, v in pieces.items()}
            )
    return {
        **mpp,
        "reduce": _REDUCE_TEMPLATE.substitute(subs),
        "reduce_raw": _REDUCE_RAW_TEMPLATE.substitute(subs),
        "matvec": _MATVEC_TEMPLATE.substitute(subs),
        "wide": _WIDE_TEMPLATE.substitute(subs),
    }


_SOURCES = {act: _sources(act) for act in ACTS}


def _lit(*values: int) -> list[IntOrVid]:
    return [IntOrVid.from_literal(v) for v in values]


@REGISTRY.register(target=[torch.ops.mlx.int8_int4_affine_gemm.default])
def _int8_int4_affine_gemm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    return _lower_int4_affine_gemm(P, n, "int8")


@REGISTRY.register(target=[torch.ops.mlx.int8sym_int4_affine_gemm.default])
def _int8sym_int4_affine_gemm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    return _lower_int4_affine_gemm(P, n, "int8sym")


@REGISTRY.register(target=[torch.ops.mlx.fp_int4_affine_gemm.default])
def _fp_int4_affine_gemm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    return _lower_int4_affine_gemm(P, n, "fp")


def _lower_int4_affine_gemm(  # noqa: C901
    P: MLXProgramBuilder, n: Node, act: str
) -> Slot:
    op_name = f"mlx::{act}_int4_affine_gemm"
    args = P.args(n)
    x_slot, w_packed_slot, w_scale_slot, w_bias_slot = args[:4]
    # Only the affine int8 op takes w_sum (for the activation zero point).
    w_sum_slot = args[4] if act == "int8" else None
    int8_act = act != "fp"
    x_node = n.args[0]
    x_meta = x_node.meta["val"]
    N, KP = n.args[1].meta["val"].shape
    w_scale_meta = n.args[2].meta["val"]
    w_bias_meta = n.args[3].meta["val"]
    G = w_scale_meta.shape[0]
    K = x_meta.shape[-1]
    if not all(isinstance(v, int) for v in (N, KP, G, K)):
        raise NotImplementedError(
            f"{op_name}: w_packed / w_scale shapes and K must be statically known"
        )
    if K != 2 * KP:
        raise ValueError(f"{op_name}: x last dim {K} != 2 * w_packed width {KP}")
    if tuple(w_scale_meta.shape) != (G, N) or w_bias_meta.shape != w_scale_meta.shape:
        raise ValueError(
            f"{op_name}: w_scale / w_bias must both be [G, N] = [{G}, {N}]; got "
            f"{tuple(w_scale_meta.shape)} / {tuple(w_bias_meta.shape)}"
        )
    if w_bias_meta.dtype != w_scale_meta.dtype:
        raise ValueError(
            f"{op_name}: w_scale and w_bias must share a dtype; got "
            f"{w_scale_meta.dtype} / {w_bias_meta.dtype}"
        )
    if K % _K_ALIGN != 0:
        raise NotImplementedError(
            f"{op_name}: K must be a multiple of {_K_ALIGN}; got K={K}"
        )
    if K % G != 0 or (K // G) % _K_ALIGN != 0:
        raise NotImplementedError(
            f"{op_name}: group size K / {G} must be a multiple of {_K_ALIGN}"
        )
    gs = K // G
    if N < 8:
        # MLX passes inputs under 8 elements in constant memory, which the
        # kernels' vector loads and MPP tensors cannot read.
        raise NotImplementedError(f"{op_name}: N must be at least 8; got N={N}")
    if act == "fp" and x_meta.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(
            f"{op_name}: activations must be fp16 or bf16; got {x_meta.dtype}"
        )

    dtype_int = torch_dtype_to_scalar_type(x_meta.dtype)
    qparam_dtype_int = torch_dtype_to_scalar_type(w_scale_meta.dtype)
    out = P.make_or_get_slot(n)
    leading = emit_shape(P, x_node, x_slot, end_dim=-1)
    out_shape_flat = leading + [IntOrVid.from_literal(N)]
    m_iov = emit_product(P, leading)
    _, xq_slot = P.make_tmp_slot()
    _, x_scale_slot = P.make_tmp_slot()
    _, x_zero_slot = P.make_tmp_slot()
    _, x_sum_slot = P.make_tmp_slot()
    _, partial_slot = P.make_tmp_slot()
    int8 = torch_dtype_to_scalar_type(torch.int8)
    int32 = torch_dtype_to_scalar_type(torch.int32)
    fp32 = torch_dtype_to_scalar_type(torch.float32)
    k_lit = IntOrVid.from_literal(K)
    n_lit = IntOrVid.from_literal(N)
    tag = f"{act}_int4_affine_gemm"
    sources = _SOURCES[act]

    # Activation-side inputs of the GEMM kernels and the inputs the int8
    # epilogues read.
    if act == "int8":
        x_inputs = {"xq": xq_slot}
        epilogue_inputs = {
            "x_scale": x_scale_slot,
            "x_zero": x_zero_slot,
            "w_sum": w_sum_slot,
        }
    elif act == "int8sym":
        x_inputs = {"xq": xq_slot}
        epilogue_inputs = {"x_scale": x_scale_slot}
    else:
        x_inputs = {"x": x_slot}
        epilogue_inputs = {}
    w_inputs = {
        "w_packed": w_packed_slot,
        "w_scale": w_scale_slot,
        "w_bias": w_bias_slot,
    }

    def emit_kernel(
        kind: str,
        inputs: dict[str, Slot],
        outputs: dict[str, tuple[Slot, int, list[IntOrVid]]],
        grid: list[IntOrVid],
        threads: int,
        template_args: dict[str, int],
        source: Optional[str] = None,
    ) -> None:
        names = list(template_args)
        P.emit(
            MetalKernelNode(
                name=f"{tag}_{kind}",
                source=sources[kind] if source is None else source,
                header=_HEADER,
                inputs=[P.slot_to_tid(s) for s in inputs.values()],
                outputs=[P.slot_to_tid(s) for s, _, _ in outputs.values()],
                grid=grid,
                threadgroup=_lit(threads, 1, 1),
                input_names=list(inputs),
                output_names=list(outputs),
                output_shapes_flat=[
                    d for _, _, shape in outputs.values() for d in shape
                ],
                output_shape_lengths=[len(shape) for _, _, shape in outputs.values()],
                output_dtypes=[dt for _, dt, _ in outputs.values()],
                template_arg_names=names,
                template_arg_kinds=[2 if a in ("T", "ST") else 0 for a in names],
                template_arg_values=list(template_args.values()),
            )
        )

    def emit_prologue(rows: IntOrVid, xg: int) -> None:
        nxg = IntOrVid.from_literal(K // xg)
        targs = {"T": dtype_int, "K": K, "TPR": _PROLOGUE_THREADS, "XG": xg}
        if int8_act:
            # rows > M zero-pads xq and x_sum to whole BM tiles; x_scale and
            # x_zero stay [M].
            quant_outputs = {
                "xq": (xq_slot, int8, [rows, k_lit]),
                "x_scale": (x_scale_slot, fp32, [m_iov]),
                "x_zero": (x_zero_slot, fp32, [m_iov]),
                "x_sum": (x_sum_slot, fp32, [rows, nxg]),
            }
            if act == "int8sym":
                del quant_outputs["x_zero"]
            emit_kernel(
                "quantize",
                {"x": x_slot},
                quant_outputs,
                [
                    IntOrVid.from_literal(_PROLOGUE_THREADS),
                    rows,
                    IntOrVid.from_literal(1),
                ],
                _PROLOGUE_THREADS,
                targs,
                source=(
                    _INT8_SYM_QUANT_SOURCE if act == "int8sym" else _INT8_QUANT_SOURCE
                ),
            )
            return
        blocks_per_tg = _PROLOGUE_THREADS // 32
        emit_kernel(
            "x_sum",
            {"x": x_slot},
            {"x_sum": (x_sum_slot, fp32, [rows, nxg])},
            [
                IntOrVid.from_literal(
                    (K // xg + blocks_per_tg - 1) // blocks_per_tg * _PROLOGUE_THREADS
                ),
                rows,
                IntOrVid.from_literal(1),
            ],
            _PROLOGUE_THREADS,
            targs,
            source=_FP_SUM_SOURCE,
        )

    def emit_matvec(mr: int) -> None:
        if int8_act:
            emit_prologue(m_iov, K)
        rows_per_tg = _MATVEC_SIMDGROUPS * _MATVEC_ROWS_PER_SIMDGROUP
        emit_kernel(
            "matvec",
            {**x_inputs, **w_inputs, **epilogue_inputs},
            {"out": (out, dtype_int, out_shape_flat)},
            [
                IntOrVid.from_literal(
                    (N + rows_per_tg - 1) // rows_per_tg * _MATVEC_SIMDGROUPS * 32
                ),
                emit_ceil_div(P, m_iov, mr),
                IntOrVid.from_literal(1),
            ],
            _MATVEC_SIMDGROUPS * 32,
            {"T": dtype_int, "N": N, "K": K, "MR": mr, "GS": gs},
        )

    def emit_wide(nv: int, kl: int) -> None:
        if K % (16 * kl):
            raise NotImplementedError(
                f"{op_name}: wide kernel needs K % {16 * kl} == 0; got K={K}"
            )
        if int8_act:
            emit_prologue(m_iov, K)
        channels_per_tg = _MATVEC_SIMDGROUPS * (32 // kl)
        emit_kernel(
            "wide",
            {**x_inputs, **w_inputs, **epilogue_inputs},
            {"out": (out, dtype_int, out_shape_flat)},
            [
                IntOrVid.from_literal(
                    (N + channels_per_tg - 1)
                    // channels_per_tg
                    * _MATVEC_SIMDGROUPS
                    * 32
                ),
                emit_ceil_div(P, m_iov, nv),
                IntOrVid.from_literal(1),
            ],
            _MATVEC_SIMDGROUPS * 32,
            {"T": dtype_int, "N": N, "K": K, "NV": nv, "KL": kl, "GS": gs},
        )

    def emit_mpp(plan: AffinePlan) -> None:
        bm, bn, splits = plan.bm, plan.bn, plan.splits
        if gs < K:
            splits = effective_splits(G, splits)
        tiles_m = emit_ceil_div(P, m_iov, bm)
        rows = emit_product(P, [tiles_m, IntOrVid.from_literal(bm)])
        emit_prologue(rows, K // splits if gs == K else gs)
        coop = plan.sm == 0
        sm, sn = (bm, bn) if coop else (plan.sm, plan.sn)
        # Weights take fixed-extent slices unless N < BN (see _MPP_TEMPLATE).
        dynw = "_dynw" if N < bn else ""
        threads = 128 if coop else (bm // sm) * (bn // sn) * 32
        grid = [
            IntOrVid.from_literal((N + bn - 1) // bn * threads),
            tiles_m,
            IntOrVid.from_literal(splits),
        ]
        # int8 kernels read M from x_scale.
        m_inputs = {"x_scale": x_scale_slot} if int8_act else {}
        mpp_inputs = {**x_inputs, "x_sum": x_sum_slot, **w_inputs, **m_inputs}
        targs = {
            "T": dtype_int,
            "ST": qparam_dtype_int,
            "N": N,
            "K": K,
            "BM": bm,
            "BN": bn,
            "SM": sm,
            "SN": sn,
            "NM": 1 if coop else _SG_ROWS_PER_THREAD.get((sm, sn), 4),
            "GS": gs,
            "SPLITS": splits,
            "BIAS": plan.bias,
            "COOP": int(coop),
        }
        if splits == 1:
            emit_kernel(
                "full_k" + dynw,
                {**mpp_inputs, **epilogue_inputs},
                {"out": (out, dtype_int, out_shape_flat)},
                grid,
                threads,
                targs,
            )
            return
        # Per-channel splits keep unscaled partials (INT32 for int8) and fold
        # once in the reduce; grouped splits sum scaled FP32 partials.
        raw = gs == K
        emit_kernel(
            "split_k" + dynw,
            mpp_inputs,
            {
                "partial": (
                    partial_slot,
                    int32 if raw and int8_act else fp32,
                    [IntOrVid.from_literal(splits), m_iov, n_lit],
                )
            },
            grid,
            threads,
            targs,
        )
        if raw:
            reduce_kind = "reduce_raw"
            reduce_inputs = {
                "partial": partial_slot,
                "x_sum": x_sum_slot,
                "w_scale": w_scale_slot,
                "w_bias": w_bias_slot,
                **epilogue_inputs,
            }
        else:
            reduce_kind = "reduce"
            reduce_inputs = {"partial": partial_slot, **epilogue_inputs}
        vw = 4 if N % 4 == 0 else 1
        emit_kernel(
            reduce_kind,
            reduce_inputs,
            {"out": (out, dtype_int, out_shape_flat)},
            [
                emit_product(
                    P,
                    [
                        emit_ceil_div(
                            P,
                            emit_product(P, [m_iov, IntOrVid.from_literal(N // vw)]),
                            256,
                        ),
                        IntOrVid.from_literal(256),
                    ],
                ),
                IntOrVid.from_literal(1),
                IntOrVid.from_literal(1),
            ],
            256,
            {"T": dtype_int, "N": N, "SPLITS": splits},
        )

    def emit_plan(plan: AffinePlan) -> None:
        if plan.bn == 0:
            emit_matvec(plan.bm)
        elif plan.bn < 0:
            emit_wide(plan.bm, -plan.bn)
        else:
            emit_mpp(plan)

    # (M - 1) // max_m == 0 exactly when M <= max_m, so each IfNode's else
    # branch takes the first plan that covers M.
    m_minus_1 = emit_sub_int(P, m_iov, IntOrVid.from_literal(1))

    def emit_plans(plans: list[AffinePlan]) -> None:
        plan, rest = plans[0], plans[1:]
        if not rest:
            emit_plan(plan)
            return
        emit_if_else(
            P,
            emit_floordiv(P, m_minus_1, IntOrVid.from_literal(plan.max_m)),
            emit_then=lambda: emit_plans(rest),
            emit_else=lambda: emit_plan(plan),
        )

    emit_plans(affine_plans(N, K, gs, act))
    return out
