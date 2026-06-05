# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Int4Tensor F.linear dispatch for CUDA — runs at eager / export trace time.

This module overrides Int4Tensor's F.linear dispatch so that torch.export
traces through our custom op and dequant logic instead of torchao's default
(mslk/tinygemm). The code here executes during eager inference and during
AOTI export tracing — it does NOT run at .pte runtime.

At .pte runtime, the captured graph is executed by the AOTI-generated .so:
  - The custom op ``executorch_cuda::int4_plain_mm`` maps to a C shim that
    runs the W4A8 dp4a matvec kernel (backends/cuda/runtime/shims/).
  - The inline dequant + F.linear is compiled by inductor into fused Triton
    dequant + cuBLAS matmul kernels.

Dispatch strategy (determines what gets captured in the export graph):
  Decode (M<=4): Custom op ``executorch_cuda::int4_plain_mm``
  Prefill (M>4): Inline dequant + F.linear (standard PyTorch ops)

The same strategy is applied to INT8 (``IntxUnpackedToInt8Tensor``) weights via
the ``executorch_cuda::int8_plain_mm`` op (W8A8 dp4a matvec). Mixed-precision
recipes (e.g. INT8 edge-layer v_proj/down_proj + INT4 elsewhere) thus keep ALL
decode linears on a fused dp4a path instead of falling back to the generic
dequant-to-bf16 + matmul path, which materializes the full weight in HBM.

Import this module before using nn.Linear with Int4Tensor / INT8 weights::

    import executorch.backends.cuda.int4_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from torch.library import impl, Library
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    IntxUnpackedToInt8Tensor,
)

# ---------------------------------------------------------------------------
# Custom op for decode (M=1): dp4a matvec in C shim, dequant+F.linear in eager
# ---------------------------------------------------------------------------

_lib = Library("executorch_cuda", "DEF")
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
# Custom op for INT8 decode (M<=4): W8A8 dp4a matvec in C shim.
#
# INT8 weights use the torchao ``IntxUnpackedToInt8Tensor`` subclass, whose
# layout differs from Int4Tensor:
#   qdata : [N, K]          int8 (one value per element, natural k order)
#   scale : [N, K//gs]      bf16 (per-group, row-major)
#   zero  : [N, K//gs]      int8 (per-group asymmetric zero point)
# vs Int4Tensor's nibble-packed [N, K//2] qdata. (For CUDA, Int4Tensor's
# scale/zero are repacked to the same coalesced [N, K//gs] layout at pack time;
# see pack_cuda.pack_linear_for_cuda.) The op signature mirrors int4_plain_mm
# for shim uniformity.
# ---------------------------------------------------------------------------

_lib.define(
    "int8_plain_mm(Tensor self, Tensor qdata, Tensor scale, Tensor zero, int group_size) -> Tensor"
)


@impl(_lib, "int8_plain_mm", "Meta")
def _meta_int8(self, qdata, scale, zero, group_size):
    return torch.empty(
        self.shape[0], qdata.shape[0], dtype=self.dtype, device=self.device
    )


@impl(_lib, "int8_plain_mm", "CUDA")
def _cuda_int8(self, qdata, scale, zero, group_size):
    return _dequant_matmul_int8(self, qdata, scale, zero, group_size)


def _dequant_matmul_int8(x, qdata, scale, zero, group_size):
    """Dequant INT8 weights to input dtype and call F.linear.

    qdata [N, K] int8, scale/zero [N, K//gs]. Per-group asymmetric:
    w[n, k] = (qdata[n, k] - zero[n, k//gs]) * scale[n, k//gs].
    """
    N, K = qdata.shape
    n_groups = K // group_size
    dtype = x.dtype

    q = qdata.to(dtype).reshape(N, n_groups, group_size)
    s = scale.to(dtype).reshape(N, n_groups, 1)
    z = zero.to(dtype).reshape(N, n_groups, 1)
    w_deq = ((q - z) * s).reshape(N, K)

    return F.linear(x, w_deq)


# ---------------------------------------------------------------------------
# Int4Tensor F.linear dispatch
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements = Int4Tensor.implements
_implements_torch_function = Int4Tensor.implements_torch_function


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
    if M <= 4:
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


# ---------------------------------------------------------------------------
# IntxUnpackedToInt8Tensor F.linear dispatch (W8A8 dp4a for decode)
# ---------------------------------------------------------------------------

_implements_i8 = IntxUnpackedToInt8Tensor.implements
_implements_torch_function_i8 = IntxUnpackedToInt8Tensor.implements_torch_function


@_implements_i8([aten.linear.default])
@_implements_torch_function_i8([F.linear])
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else None

    # Only the weight-only INT8 (target_dtype=int8) case is routed through the
    # fused dp4a path. Anything else (e.g. dynamic activation quant, non-int8
    # target_dtype used by other backends) falls back to the generic dequant.
    if (
        weight_tensor.target_dtype is not torch.int8
        or weight_tensor.activation_quantization is not None
    ):
        return F.linear(input_tensor, weight_tensor.dequantize(), bias)

    orig_shape = input_tensor.shape
    x_2d = input_tensor.reshape(-1, orig_shape[-1])

    qdata = weight_tensor.qdata
    scale = weight_tensor.scale
    zero = weight_tensor.zero_point
    gs = weight_tensor.block_size[-1]

    M = x_2d.shape[0]
    if M <= 4:
        out = torch.ops.executorch_cuda.int8_plain_mm(x_2d, qdata, scale, zero, gs)
    else:
        out = _dequant_matmul_int8(x_2d, qdata, scale, zero, gs)

    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out
