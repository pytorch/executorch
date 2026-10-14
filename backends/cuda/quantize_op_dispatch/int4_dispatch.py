# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CudaCoalescedInt4Tensor F.linear dispatch for CUDA — runs at eager / export trace time.

This module registers an F.linear dispatch on ``CudaCoalescedInt4Tensor`` (an
ExecuTorch-internal subclass, see ``coalesced_int4_tensor.py``) so that
torch.export traces through our Triton ops and dequant logic. Routing is by
*type*: stock torchao ``Int4Tensor`` weights are left untouched and keep using
torchao's default (mslk/tinygemm) path. The code here executes during eager
inference and during AOTI export tracing — it does NOT run at .pte runtime.

At .pte runtime, AOTInductor executes the captured kernels. Supported BF16
inputs use W4A8 DP4A for M <= 4, fused W4A16 kernels for M <= 64, and
Q8-activation tensor-core MMQ for M <= 4096. Weights keep their packed storage.
Dynamic M uses the smallest bucket whose bounds cover the input. Unsupported
shapes, group sizes, dtypes and CPU eager use dequant + F.linear.

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the other formats) before using nn.Linear with
CudaCoalescedInt4Tensor weights::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    chunked_dequant_linear,
    quantized_linear,
)


def _dequant_matmul(x, qdata, scale, scale_step, zero, zero_point_step, group_size):
    """Dequant INT4 weights to input dtype and call F.linear.

    Metadata is in the coalesced [N, n_groups] layout (baked into the weight
    constant at pack time), aligned row-for-row with qdata's [N, *]. The scale is
    a uint8 code with a per-256-super-block fp16 ``scale_step`` ([N, K/256]); the
    real per-group scale is ``scale_code * scale_step[:, g // 8]``. The zero is a
    uint8 code with a per-256-super-block fp16 ``zero_point_step`` ([N, K/256]);
    the real per-group zero is ``zero_code * zero_point_step[:, g // 8]``.
    """
    N, K_half = qdata.shape
    K = K_half * 2
    n_groups = K // group_size
    gs_half = group_size // 2
    n_super = K // 256
    groups_per_super = n_groups // n_super
    dtype = x.dtype

    def dequant_linear_rows(i, j):
        rows = j - i
        p = qdata[i:j].to(torch.uint8).reshape(rows, n_groups, gs_half)
        low = (p & 0x0F).to(dtype)
        high = ((p >> 4) & 0x0F).to(dtype)
        data = torch.stack([low, high], dim=-1).reshape(rows, n_groups, group_size)
        # Scale and zero: uint8 code * per-256 fp16 step (broadcast over the
        # groups in each super-block).
        s = (
            scale[i:j].to(dtype)
            * scale_step[i:j].to(dtype).repeat_interleave(groups_per_super, dim=1)
        ).unsqueeze(-1)
        z = (
            zero[i:j].to(dtype)
            * zero_point_step[i:j].to(dtype).repeat_interleave(groups_per_super, dim=1)
        ).unsqueeze(-1)
        w_deq = ((data - z) * s).reshape(rows, K)
        return F.linear(x, w_deq)

    return chunked_dequant_linear(x, N, dequant_linear_rows)


# ---------------------------------------------------------------------------
# CudaCoalescedInt4Tensor F.linear dispatch
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements = CudaCoalescedInt4Tensor.implements
_implements_torch_function = CudaCoalescedInt4Tensor.implements_torch_function


@_implements([aten.linear.default])
@_implements_torch_function([F.linear])
def _(func, types, args, kwargs):
    from executorch.backends.cuda.triton.kernels.int4_quantized_gemm import (
        INT4_QUANTIZED_GEMM,
    )

    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
    # The metadata is already in the coalesced [N, n_groups] layout the kernels
    # read, so it passes straight through.
    weight_args = (
        weight.qdata,
        weight.scale,
        weight.scale_step,
        weight.zero_point,
        weight.zero_point_step,
        weight.block_size[-1],
    )
    return quantized_linear(
        INT4_QUANTIZED_GEMM,
        input_tensor,
        weight_args,
        bias,
        lambda x_2d: _dequant_matmul(x_2d, *weight_args),
        max_bucket=4096,
    )
