# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaDp4aPlanarInt6Tensor F.linear dispatch for CUDA — eager / export trace time.

This module registers an F.linear dispatch on ``CudaDp4aPlanarInt6Tensor`` (an
ExecuTorch-internal subclass, see ``dp4a_planar_int6_tensor.py``) so that
torch.export traces through our custom op and dequant logic. Routing is by
*type*: only GGUF Q6_K weights (converted to ``CudaDp4aPlanarInt6Tensor``) take the
packed-int6 path; genuine INT8 weights stay on the int8 path. The code here runs
during eager inference and AOTI export tracing — it does NOT run at .pte runtime.

At .pte runtime, the captured graph is executed by the AOTI-generated .so:
  - The custom op ``executorch_cuda::int6_plain_mm`` maps to a C shim that runs
    the W6A8 dp4a matvec kernel (backends/cuda/runtime/shims/int6_plain_mm.*).
  - The inline dequant + F.linear is compiled by inductor into fused Triton
    dequant + cuBLAS matmul kernels.

Dispatch strategy (determines what gets captured in the export graph):
  Decode (M<=4): Custom op ``executorch_cuda::int6_plain_mm``
  Prefill (M>4): Inline dequant + F.linear (standard PyTorch ops)

The packed-int6 weight is symmetric (no zero point): ``w = q * scale`` with
``q`` in ``[-32, 31]`` stored as the ql/qh planes. The op signature mirrors
int4_plain_mm / int8_plain_mm but takes two weight planes (ql, qh) instead of
one, and no zero tensor.

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the INT4 / INT8 ones)::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.dp4a_planar_int6_tensor import (
    CudaDp4aPlanarInt6Tensor,
    unpack_int6,
)
from executorch.backends.cuda.quantize_op_dispatch._library import lib as _lib
from torch.library import impl

# ---------------------------------------------------------------------------
# Custom op for INT6 decode (M<=4): W6A8 dp4a matvec in C shim.
# ---------------------------------------------------------------------------

_lib.define(
    "int6_plain_mm(Tensor self, Tensor ql, Tensor qh, Tensor scale, Tensor steps, int group_size) -> Tensor"
)


@impl(_lib, "int6_plain_mm", "Meta")
def _meta_int6(self, ql, qh, scale, steps, group_size):
    return torch.empty(self.shape[0], ql.shape[0], dtype=self.dtype, device=self.device)


@impl(_lib, "int6_plain_mm", "CUDA")
def _cuda_int6(self, ql, qh, scale, steps, group_size):
    # scale is int8 codes in the [N, n_groups] layout; steps is the per-256
    # super-block [N, K/256] fp16 scale step. _unit_dq_mm_int6 reconstructs
    # scale = code * steps[:, g // (256 // gs)].
    return _unit_dq_mm_int6(self, ql, qh, scale, steps, group_size)


def _unit_dq_mm_int6(x, ql, qh, scale, steps, group_size):
    """Dequant packed-INT6 weights to input dtype and call F.linear.

    ql [N, K/2] / qh [N, K/4] pack symmetric Q6_K values q in [-32, 31].
    scale [N, K//gs] is int8 codes; steps [N, K//256] fp16 is the per-256
    super-block scale step, so the real per-group scale is
    ``scale_code * steps[:, g // (256 // gs)]``. Dequant:
    w[n, k] = q[n, k] * (scale_code[n, k//gs] * steps[n, (k//gs) // gps]).
    """
    N = ql.shape[0]
    K = ql.shape[1] * 2
    n_groups = K // group_size
    n_super = steps.shape[1]
    groups_per_super = n_groups // n_super
    dtype = x.dtype

    q = unpack_int6(ql, qh, N, K).to(dtype).reshape(N, n_groups, group_size)
    # Broadcast the per-256 step over the groups_per_super groups in each
    # super-block, then multiply by the int8 code -> effective per-group scale.
    step_g = steps.to(dtype).repeat_interleave(groups_per_super, dim=1)
    s = (scale.to(dtype) * step_g).reshape(N, n_groups, 1)
    w_deq = (q * s).reshape(N, K)

    return F.linear(x, w_deq)


# ---------------------------------------------------------------------------
# CudaDp4aPlanarInt6Tensor F.linear dispatch (W6A8 dp4a for decode)
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements_i6 = CudaDp4aPlanarInt6Tensor.implements
_implements_torch_function_i6 = CudaDp4aPlanarInt6Tensor.implements_torch_function


@_implements_i6([aten.linear.default])
@_implements_torch_function_i6([F.linear])
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight_tensor = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)

    orig_shape = input_tensor.shape
    x_2d = input_tensor.reshape(-1, orig_shape[-1])

    ql = weight_tensor.ql
    qh = weight_tensor.qh
    scale = weight_tensor.scale
    steps = weight_tensor.steps
    gs = weight_tensor.block_size[-1]

    M = x_2d.shape[0]
    if M <= 4:
        out = torch.ops.executorch_cuda.int6_plain_mm(x_2d, ql, qh, scale, steps, gs)
    else:
        out = _unit_dq_mm_int6(x_2d, ql, qh, scale, steps, gs)

    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out
