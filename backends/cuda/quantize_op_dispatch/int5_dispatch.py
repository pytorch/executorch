# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaDp4aPlanarInt5Tensor F.linear dispatch for CUDA — eager / export trace time.

This module registers an F.linear dispatch on ``CudaDp4aPlanarInt5Tensor`` (an
ExecuTorch-internal subclass, see ``dp4a_planar_int5_tensor.py``) so that
torch.export traces through our custom op and dequant logic. Routing is by
*type*: only GGUF Q5_K weights (converted to ``CudaDp4aPlanarInt5Tensor``) take
the packed-int5 path. The code here runs during eager inference and AOTI export
tracing — it does NOT run at .pte runtime.

At .pte runtime, the captured graph is executed by the AOTI-generated .so:
  - The custom op ``executorch_cuda::int5_plain_mm`` maps to a C shim that runs
    the W5A8 dp4a matvec kernel (backends/cuda/runtime/shims/int5_plain_mm.*).
  - The inline dequant + F.linear is compiled by inductor into fused Triton
    dequant + cuBLAS matmul kernels.

Dispatch strategy (determines what gets captured in the export graph):
  Decode (M<=4): Custom op ``executorch_cuda::int5_plain_mm``
  Prefill (M>4): Inline dequant + F.linear (standard PyTorch ops)

The packed-int5 weight is asymmetric (has a zero point, like INT4): ``w =
scale * (u - zero)`` with ``u`` in ``[0, 31]`` stored as the ql/qh planes. The op
signature mirrors int4_plain_mm: two weight planes (ql, qh) plus uint8 scale +
per-256 fp16 scale_step + uint8 zero + per-256 fp16 zero_step (z_pack).

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the INT4 / INT6 / INT8 ones)::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.dp4a_planar_int5_tensor import (
    CudaDp4aPlanarInt5Tensor,
    unpack_int5,
)
from executorch.backends.cuda.quantize_op_dispatch._library import lib as _lib
from torch.library import impl

# ---------------------------------------------------------------------------
# Custom op for INT5 decode (M<=4): W5A8 dp4a matvec in C shim.
# ---------------------------------------------------------------------------

_lib.define(
    "int5_plain_mm(Tensor self, Tensor ql, Tensor qh, Tensor scale, Tensor scale_step, Tensor zero, Tensor zero_step, int group_size) -> Tensor"
)


@impl(_lib, "int5_plain_mm", "Meta")
def _meta_int5(self, ql, qh, scale, scale_step, zero, zero_step, group_size):
    return torch.empty(self.shape[0], ql.shape[0], dtype=self.dtype, device=self.device)


@impl(_lib, "int5_plain_mm", "CUDA")
def _cuda_int5(self, ql, qh, scale, scale_step, zero, zero_step, group_size):
    # scale/zero are uint8 codes in the [N, n_groups] layout; scale_step and
    # zero_step are per-256-super-block [N, K/256] fp16 steps (z_pack).
    # _dequant_matmul_int5 reconstructs scale=code*scale_step[g//8],
    # zero=code*zero_step[g//8].
    return _dequant_matmul_int5(
        self, ql, qh, scale, scale_step, zero, zero_step, group_size
    )


# Chunked dequant for the export GPU budget (see int4_dispatch for the rationale;
# the lm_head is the only weight that crosses the threshold, and only via the
# custom-op path — never the runtime graph, so ZERO runtime / accuracy impact).
_DEQUANT_N_THRESHOLD = 65536
_DEQUANT_N_CHUNK = 32768


def _dequant_matmul_int5(x, ql, qh, scale, scale_step, zero, zero_step, group_size):
    """Dequant packed-INT5 weights to input dtype and call F.linear.

    ql [N, K/2] / qh [N, K/8] pack asymmetric Q5_K values u in [0, 31].
    scale/zero [N, K//gs] are uint8 codes; scale_step / zero_step [N, K/256] are
    per-256-super-block fp16 steps, so the real per-group values are
    ``scale = scale_code * scale_step[:, g // 8]`` and ``zero = zero_code *
    zero_step[:, g // 8]`` (z_pack, mirroring INT4). Dequant:
    w[n, k] = scale[n, k//gs] * (u[n, k] - zero[n, k//gs]).
    """
    N = ql.shape[0]
    K = ql.shape[1] * 2
    n_groups = K // group_size
    n_super = K // 256
    groups_per_super = n_groups // n_super
    dtype = x.dtype

    def _dq(qlc, qhc, sc, s_step, ze, z_step, rows):
        u = unpack_int5(qlc, qhc, rows, K).to(dtype).reshape(rows, n_groups, group_size)
        # Scale/zero: uint8 code * per-256 fp16 step (broadcast over the 8 groups
        # in each super-block).
        scale_step_g = s_step.to(dtype).repeat_interleave(groups_per_super, dim=1)
        s = (sc.to(dtype) * scale_step_g).reshape(rows, n_groups, 1)
        zero_step_g = z_step.to(dtype).repeat_interleave(groups_per_super, dim=1)
        z = (ze.to(dtype) * zero_step_g).reshape(rows, n_groups, 1)
        w_deq = (s * (u - z)).reshape(rows, K)
        return F.linear(x, w_deq)

    if N <= _DEQUANT_N_THRESHOLD:
        return _dq(ql, qh, scale, scale_step, zero, zero_step, N)

    outs = []
    for i in range(0, N, _DEQUANT_N_CHUNK):
        j = min(i + _DEQUANT_N_CHUNK, N)
        outs.append(
            _dq(
                ql[i:j],
                qh[i:j],
                scale[i:j],
                scale_step[i:j],
                zero[i:j],
                zero_step[i:j],
                j - i,
            )
        )
    return torch.cat(outs, dim=-1)


# ---------------------------------------------------------------------------
# CudaDp4aPlanarInt5Tensor F.linear dispatch (W5A8 dp4a for decode)
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements_i5 = CudaDp4aPlanarInt5Tensor.implements
_implements_torch_function_i5 = CudaDp4aPlanarInt5Tensor.implements_torch_function


@_implements_i5([aten.linear.default])
@_implements_torch_function_i5([F.linear])
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)

    orig_shape = input_tensor.shape
    x_2d = input_tensor.reshape(-1, orig_shape[-1])

    ql = weight_tensor.ql
    qh = weight_tensor.qh
    scale = weight_tensor.scale
    scale_step = weight_tensor.scale_step
    zero = weight_tensor.zero_point
    zero_step = weight_tensor.zero_step
    gs = weight_tensor.block_size[-1]

    M = x_2d.shape[0]
    if M <= 4:
        out = torch.ops.executorch_cuda.int5_plain_mm(
            x_2d, ql, qh, scale, scale_step, zero, zero_step, gs
        )
    else:
        out = _dequant_matmul_int5(
            x_2d, ql, qh, scale, scale_step, zero, zero_step, gs
        )

    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out
