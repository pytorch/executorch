# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaDp4aPlanarInt5Tensor F.linear dispatch for CUDA — eager / export trace time.

This module registers an F.linear dispatch on ``CudaDp4aPlanarInt5Tensor`` (an
ExecuTorch-internal subclass, see ``dp4a_planar_int5_tensor.py``) so that
torch.export traces through our Triton ops and dequant logic. Routing is by
*type*: only GGUF Q5_K weights (converted to ``CudaDp4aPlanarInt5Tensor``) take
the packed-int5 path. The code here runs during eager inference and AOTI export
tracing — it does NOT run at .pte runtime.

At .pte runtime, the captured graph is executed by the AOTI-generated .so:
  - ``triton::int5_quantized_gemm_m{M}`` is a Triton W5A8 DP4A kernel compiled
    into it (see triton/kernels/int5_quantized_gemm.py).
  - The inline dequant + F.linear is compiled by inductor into fused Triton
    dequant + matmul kernels.

Dispatch (``_gemm_family_dispatch.quantized_linear``): when a Triton kernel can
run, the smallest bucket of ``INT5_QUANTIZED_GEMM`` that supports the inputs
(static M <= 4 takes its own bucket; a dynamic M provably within [1, 4] takes
the smallest bucket that holds it). Everything else (prefill, an unbounded
dynamic M, other group sizes or dtypes, CPU eager) uses inline dequant +
F.linear, never an error.

The packed-int5 weight is asymmetric (has a zero point, like INT4): ``w =
scale * (u - zero)`` with ``u`` in ``[0, 31]`` stored as the ql/qh planes, and
uint8 scale/zero codes with per-256 fp16 steps.

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the other formats)::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.dp4a_planar_int5_tensor import (
    CudaDp4aPlanarInt5Tensor,
    unpack_int5,
)
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    chunked_dequant_linear,
    quantized_linear,
)


def _dequant_matmul_int5(
    x, ql, qh, scale, scale_step, zero, zero_point_step, group_size
):
    """Dequant packed-INT5 weights to input dtype and call F.linear.

    ql [N, K/2] / qh [N, K/8] pack asymmetric Q5_K values u in [0, 31].
    scale/zero [N, K//gs] are uint8 codes; scale_step / zero_point_step [N, K/256] are
    per-256-super-block fp16 steps, so the real per-group values are
    ``scale = scale_code * scale_step[:, g // 8]`` and ``zero = zero_code *
    zero_point_step[:, g // 8]`` (z_pack, mirroring INT4). Dequant:
    w[n, k] = scale[n, k//gs] * (u[n, k] - zero[n, k//gs]).
    """
    N = ql.shape[0]
    K = ql.shape[1] * 2
    n_groups = K // group_size
    n_super = K // 256
    groups_per_super = n_groups // n_super
    dtype = x.dtype

    def dequant_linear_rows(i, j):
        rows = j - i
        u = unpack_int5(ql[i:j], qh[i:j], rows, K).to(dtype).reshape(rows, n_groups, group_size)
        # Scale/zero: uint8 code * per-256 fp16 step (broadcast over the groups
        # in each super-block).
        s = (
            scale[i:j].to(dtype)
            * scale_step[i:j].to(dtype).repeat_interleave(groups_per_super, dim=1)
        ).reshape(rows, n_groups, 1)
        z = (
            zero[i:j].to(dtype)
            * zero_point_step[i:j].to(dtype).repeat_interleave(groups_per_super, dim=1)
        ).reshape(rows, n_groups, 1)
        return F.linear(x, (s * (u - z)).reshape(rows, K))

    return chunked_dequant_linear(x, N, dequant_linear_rows)


# ---------------------------------------------------------------------------
# CudaDp4aPlanarInt5Tensor F.linear dispatch
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements_i5 = CudaDp4aPlanarInt5Tensor.implements
_implements_torch_function_i5 = CudaDp4aPlanarInt5Tensor.implements_torch_function


@_implements_i5([aten.linear.default])
@_implements_torch_function_i5([F.linear])
def _(func, types, args, kwargs):
    from executorch.backends.cuda.triton.kernels.int5_quantized_gemm import (
        INT5_QUANTIZED_GEMM,
    )

    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
    weight_args = (
        weight.ql,
        weight.qh,
        weight.scale,
        weight.scale_step,
        weight.zero_point,
        weight.zero_point_step,
        weight.block_size[-1],
    )
    return quantized_linear(
        INT5_QUANTIZED_GEMM,
        input_tensor,
        weight_args,
        bias,
        lambda x_2d: _dequant_matmul_int5(x_2d, *weight_args),
    )
