# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaDp4aPlanarInt6Tensor F.linear dispatch for CUDA — eager / export trace time.

This module registers an F.linear dispatch on ``CudaDp4aPlanarInt6Tensor`` (an
ExecuTorch-internal subclass, see ``dp4a_planar_int6_tensor.py``) so that
torch.export traces through our Triton ops and dequant logic. Routing is by
*type*: only GGUF Q6_K weights (converted to ``CudaDp4aPlanarInt6Tensor``) take the
packed-int6 path; genuine INT8 weights stay on the int8 path. The code here runs
during eager inference and AOTI export tracing — it does NOT run at .pte runtime.

At .pte runtime, the captured graph is executed by the AOTI-generated .so:
  - ``triton::int6_quantized_gemm_m{M}`` is a Triton W6A8 DP4A kernel compiled
    into it (see triton/kernels/int6_quantized_gemm.py).
  - The inline dequant + F.linear is compiled by inductor into fused Triton
    dequant + matmul kernels.

Dispatch (``_gemm_family_dispatch.quantized_linear``): when a Triton kernel can
run, the smallest bucket of ``INT6_QUANTIZED_GEMM`` that supports the inputs
(static M <= 4 takes its own bucket; a dynamic M provably within [1, 4] takes
the smallest bucket that holds it). Everything else (prefill, an unbounded
dynamic M, other group sizes or dtypes, CPU eager) uses inline dequant +
F.linear, never an error.

The packed-int6 weight is symmetric (no zero point): ``w = q * scale`` with
``q`` in ``[-32, 31]`` stored as the ql/qh planes.

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the other formats)::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.dp4a_planar_int6_tensor import (
    CudaDp4aPlanarInt6Tensor,
    unpack_int6,
)
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    chunked_dequant_linear,
    quantized_linear,
)


def _unit_dq_mm_int6(x, ql, qh, scale, steps, group_size):
    """Dequant packed-INT6 weights to input dtype and call F.linear.

    ql [N, K/2] / qh [N, K/4] pack symmetric Q6_K values q in [-32, 31].
    scale [N, K//gs] is signed 8-bit codes (raw uint8 storage is read as
    int8, as the kernels do); steps [N, K//256] fp16 is the per-256
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
    codes = scale.view(torch.int8) if scale.dtype == torch.uint8 else scale

    def dequant_linear_rows(i, j):
        rows = j - i
        q = unpack_int6(ql[i:j], qh[i:j], rows, K).to(dtype).reshape(rows, n_groups, group_size)
        # Broadcast the per-256 step over the groups in each super-block, then
        # multiply by the int8 code -> effective per-group scale.
        step_g = steps[i:j].to(dtype).repeat_interleave(groups_per_super, dim=1)
        s = (codes[i:j].to(dtype) * step_g).reshape(rows, n_groups, 1)
        return F.linear(x, (q * s).reshape(rows, K))

    return chunked_dequant_linear(x, N, dequant_linear_rows)


# ---------------------------------------------------------------------------
# CudaDp4aPlanarInt6Tensor F.linear dispatch
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements_i6 = CudaDp4aPlanarInt6Tensor.implements
_implements_torch_function_i6 = CudaDp4aPlanarInt6Tensor.implements_torch_function


@_implements_i6([aten.linear.default])
@_implements_torch_function_i6([F.linear])
def _(func, types, args, kwargs):
    from executorch.backends.cuda.triton.kernels.int6_quantized_gemm import (
        INT6_QUANTIZED_GEMM,
    )

    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
    weight_args = (
        weight.ql,
        weight.qh,
        weight.scale,
        weight.steps,
        weight.block_size[-1],
    )
    return quantized_linear(
        INT6_QUANTIZED_GEMM,
        input_tensor,
        weight_args,
        bias,
        lambda x_2d: _unit_dq_mm_int6(x_2d, *weight_args),
    )
