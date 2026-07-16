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
  Decode (M<=4): Custom op ``executorch_cuda::int4_plain_mm``
  Prefill (M>4): Inline dequant + F.linear (standard PyTorch ops)

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

_lib.define(
    "int4_plain_mm(Tensor self, Tensor qdata, Tensor scale, Tensor scale_step, Tensor zero, Tensor zero_point_step, int group_size) -> Tensor"
)


@impl(_lib, "int4_plain_mm", "Meta")
def _meta(self, qdata, scale, scale_step, zero, zero_point_step, group_size):
    return torch.empty(
        self.shape[0], qdata.shape[0], dtype=self.dtype, device=self.device
    )


@impl(_lib, "int4_plain_mm", "CUDA")
def _cuda(self, qdata, scale, scale_step, zero, zero_point_step, group_size):
    # Metadata is stored in the coalesced [N, n_groups] layout (transposed at
    # pack time, see pack_cuda.pack_linear_for_cuda). The scale is a uint8 code
    # with a per-256 fp16 scale_step; the zero is a uint8 code with a per-256
    # fp16 zero_point_step. _dequant_matmul reconstructs scale =
    # code*scale_step[g//8], zero = code*zero_point_step[g//8].
    return _dequant_matmul(
        self, qdata, scale, scale_step, zero, zero_point_step, group_size
    )


# Chunked dequant for the export GPU budget. The lm_head dequant (N = vocab_size,
# e.g. 262144) runs through the int4_plain_mm custom op (M=1); AOTI executes that
# op's CUDA impl during autotune / cpp_wrapper codegen, where it transiently holds
# ~5 full-size bf16 temporaries (low/high/data/data-z/w_deq) — ~10 GiB for a
# 262144-row weight even though the final w_deq is only ~2.6 GiB. Chunking along N
# caps that at ~chunk rows. It is numerically identical (F.linear output rows are
# independent), and because only the lm_head (custom-op) path crosses the N
# threshold — never the M>4 prefill inline path — it never enters the runtime
# graph: ZERO runtime / accuracy impact. Applied unconditionally to any weight
# whose row count exceeds the threshold.
_DEQUANT_N_THRESHOLD = 65536
_DEQUANT_N_CHUNK = 32768


def _dequant_matmul(x, qdata, scale, scale_step, zero, zero_point_step, group_size):
    """Dequant INT4 weights to input dtype and call F.linear.

    Metadata is in the coalesced [N, n_groups] layout (baked into the weight
    constant at pack time), aligned row-for-row with qdata's [N, *]. The scale is
    a uint8 code with a per-256-super-block fp16 ``scale_step`` ([N, K/256]); the
    real per-group scale is ``scale_code * scale_step[:, g // 8]``. The zero is a
    uint8 code with a per-256-super-block fp16 ``zero_point_step`` ([N, K/256]);
    the real per-group zero is ``zero_code * zero_point_step[:, g // 8]``.

    Large weights (N > threshold, i.e. the lm_head) are chunked along N to bound
    the dequant intermediate (see note above); smaller weights take the original
    single-shot dequant.
    """
    N, K_half = qdata.shape
    K = K_half * 2
    n_groups = K // group_size
    gs_half = group_size // 2
    n_super = K // 256
    groups_per_super = n_groups // n_super
    dtype = x.dtype

    def _unit_dq_mm(qd, sc, s_step, ze, z_step, rows):
        p = qd.to(torch.uint8).reshape(rows, n_groups, gs_half)
        low = (p & 0x0F).to(dtype)
        high = ((p >> 4) & 0x0F).to(dtype)
        data = torch.stack([low, high], dim=-1).reshape(rows, n_groups, group_size)
        # Scale: uint8 code * per-256 fp16 step (broadcast over the 8 groups in
        # each super-block).
        scale_step_g = s_step.to(dtype).repeat_interleave(groups_per_super, dim=1)
        s = (sc.to(dtype) * scale_step_g).unsqueeze(-1)
        # Zero: uint8 code * per-256 fp16 step (broadcast over the 8 groups in
        # each super-block).
        zero_point_step_g = z_step.to(dtype).repeat_interleave(groups_per_super, dim=1)
        z = (ze.to(dtype) * zero_point_step_g).unsqueeze(-1)
        w_deq = ((data - z) * s).reshape(rows, K)
        return F.linear(x, w_deq)

    if N <= _DEQUANT_N_THRESHOLD:
        return _unit_dq_mm(qdata, scale, scale_step, zero, zero_point_step, N)

    outs = []
    for i in range(0, N, _DEQUANT_N_CHUNK):
        j = min(i + _DEQUANT_N_CHUNK, N)
        outs.append(
            _unit_dq_mm(
                qdata[i:j],
                scale[i:j],
                scale_step[i:j],
                zero[i:j],
                zero_point_step[i:j],
                j - i,
            )
        )
    return torch.cat(outs, dim=-1)


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
    scale_step = weight_tensor.scale_step
    zero = weight_tensor.zero_point
    zero_point_step = weight_tensor.zero_point_step
    gs = weight_tensor.block_size[-1]

    M = x_2d.shape[0]
    if M <= 4:
        # The metadata is already in the coalesced [N, n_groups] layout the
        # decode kernel reads directly (baked into the weight constant at pack
        # time): scale as uint8 codes + per-256 fp16 scale_step; zero as uint8
        # codes + per-256 fp16 zero_point_step. Passing them straight through
        # keeps the export graph free of any per-step transpose/clone, so the
        # coalesced layout is realized without recomputing it every decode step.
        out = torch.ops.executorch_cuda.int4_plain_mm(
            x_2d, qdata, scale, scale_step, zero, zero_point_step, gs
        )
    else:
        out = _dequant_matmul(x_2d, qdata, scale, scale_step, zero, zero_point_step, gs)

    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out
