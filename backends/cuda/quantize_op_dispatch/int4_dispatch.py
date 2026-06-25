# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaCoalescedInt4Tensor F.linear dispatch for CUDA — runs at eager / export trace time.

This module registers an F.linear dispatch on ``CudaCoalescedInt4Tensor`` (an
ExecuTorch-internal subclass, see ``coalesced_int4_tensor.py``) so that
torch.export traces through our custom op and dequant logic. Routing is by
*type*: stock torchao ``Int4Tensor`` weights are left untouched and keep using
torchao's default (mslk/tinygemm) path. The code here executes during eager
inference and during AOTI export tracing — it does NOT run at .pte runtime.

At .pte runtime, the captured graph is executed by the AOTI-generated .so:
  - The custom op ``executorch_cuda::int4_plain_mm`` maps to a C shim that
    runs the W4A8 dp4a matvec kernel (backends/cuda/runtime/shims/).
  - The inline dequant + F.linear is compiled by inductor into fused Triton
    dequant + cuBLAS matmul kernels.

Dispatch strategy (determines what gets captured in the export graph):
  Small M (M<=MATVEC_MAX_M): Custom op ``executorch_cuda::int4_plain_mm``
  Large M (M>MATVEC_MAX_M):  Inline dequant + F.linear (standard PyTorch ops)

The custom op is memory-bound and beats dequant+cuBLAS for small M (M==1 matvec;
2<=M<=8 weight-stationary GEMM). ``MATVEC_MAX_M`` defaults to 4 (decode only).
An export may raise it up to the shim's GEMM limit (``GEMM_MAX_M`` = 8 in
``int4_plain_mm.cuh``), but then its *dynamic* shapes must not straddle the
threshold: a dynamic linear whose M range crosses MATVEC_MAX_M makes
torch.export's branch guard ambiguous, so a long-prefill export must declare
``min > MATVEC_MAX_M``. Raising the global default would break exports whose
dynamic prefill range starts below the threshold, so callers set it locally
instead.

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the INT8 one) before using nn.Linear with
CudaCoalescedInt4Tensor weights::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.backends.cuda.quantize_op_dispatch._library import lib as _lib
from torch.library import impl

# ---------------------------------------------------------------------------
# Custom op for decode (M=1): dp4a matvec in C shim, dequant+F.linear in eager
# ---------------------------------------------------------------------------

# Largest M the C++ shim's GEMM kernel handles (GEMM_MAX_M in int4_plain_mm.cuh).
# MATVEC_MAX_M must not exceed it, else export captures a shape the shim rejects
# at runtime; the dispatch asserts this below.
SHIM_GEMM_MAX_M = 8

# Max M routed to the custom INT4 op; above this, dequant+cuBLAS wins. Defaults
# to 4 (decode); an export may raise it (<= SHIM_GEMM_MAX_M) for small-M GEMM,
# subject to the dynamic-shape constraint documented above.
MATVEC_MAX_M = 4

_lib.define(
    "int4_plain_mm(Tensor self, Tensor qdata, Tensor scale, Tensor zero, int group_size) -> Tensor"
)


@impl(_lib, "int4_plain_mm", "Meta")
def _meta(self, qdata, scale, zero, group_size):
    return torch.empty(
        self.shape[0], qdata.shape[0], dtype=self.dtype, device=self.device
    )


@impl(_lib, "int4_plain_mm", "CUDA")
def _cuda(self, qdata, scale, zero, group_size):
    # scale/zero are stored in the coalesced [N, n_groups] layout (transposed
    # at pack time, see pack_cuda.pack_linear_for_cuda), which is exactly what
    # _dequant_matmul expects.
    return _dequant_matmul(self, qdata, scale, zero, group_size)


def _dequant_matmul(x, qdata, scale, zero, group_size):
    """Dequant INT4 weights to input dtype and call F.linear.

    scale/zero are in the coalesced [N, n_groups] layout (baked into the
    weight constant at pack time), aligned row-for-row with qdata's [N, *].
    """
    N, K_half = qdata.shape
    K = K_half * 2
    n_groups = K // group_size
    gs_half = group_size // 2
    dtype = x.dtype

    p = qdata.to(torch.uint8).reshape(N, n_groups, gs_half)
    low = (p & 0x0F).to(dtype)
    high = ((p >> 4) & 0x0F).to(dtype)
    data = torch.stack([low, high], dim=-1).reshape(N, n_groups, group_size)

    s = scale.to(dtype).unsqueeze(-1)
    z = zero.to(dtype).unsqueeze(-1)
    w_deq = ((data - z) * s).reshape(N, K)

    return F.linear(x, w_deq)


# ---------------------------------------------------------------------------
# CudaCoalescedInt4Tensor F.linear dispatch
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements = CudaCoalescedInt4Tensor.implements
_implements_torch_function = CudaCoalescedInt4Tensor.implements_torch_function


@_implements([aten.linear.default])
@_implements_torch_function([F.linear])
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None

    orig_shape = input_tensor.shape
    x_2d = input_tensor.reshape(-1, orig_shape[-1])

    qdata = weight_tensor.qdata
    scale = weight_tensor.scale
    zero = weight_tensor.zero_point
    gs = weight_tensor.block_size[-1]

    M = x_2d.shape[0]
    if M <= MATVEC_MAX_M:
        assert MATVEC_MAX_M <= SHIM_GEMM_MAX_M, (
            f"MATVEC_MAX_M={MATVEC_MAX_M} exceeds the shim's GEMM_MAX_M="
            f"{SHIM_GEMM_MAX_M} (int4_plain_mm.cuh)"
        )
        # scale/zero are already in the coalesced [N, n_groups] layout the
        # decode kernel reads directly (baked into the weight constant at pack
        # time). Passing them straight through keeps the export graph free of
        # any per-step transpose/clone, so the coalesced layout is realized
        # without recomputing it every decode step.
        out = torch.ops.executorch_cuda.int4_plain_mm(x_2d, qdata, scale, zero, gs)
    else:
        out = _dequant_matmul(x_2d, qdata, scale, zero, gs)

    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out
