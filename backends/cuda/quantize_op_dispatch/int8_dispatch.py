# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""IntxUnpackedToInt8Tensor F.linear dispatch for CUDA — eager / export trace time.

This module overrides ``IntxUnpackedToInt8Tensor``'s F.linear dispatch so that
torch.export traces through our Triton ops and dequant logic instead of
torchao's default. Like the INT4 path, the code here runs during eager inference
and AOTI export tracing — it does NOT run at .pte runtime.

At .pte runtime, the captured graph is executed by the AOTI-generated .so:
  - ``triton::int8_quantized_gemm_m{M}`` is a Triton W8A8 DP4A kernel compiled
    into it (see triton/kernels/int8_quantized_gemm.py).
  - The inline dequant + F.linear is compiled by inductor into fused Triton
    dequant + matmul kernels.

Only weight-only INT8 (``target_dtype`` int8, no activation quantization) takes
this path; anything else uses the tensor's own dequantize + F.linear. Within it
(``_gemm_family_dispatch.quantized_linear``): when a Triton kernel can run, the
smallest bucket of ``INT8_QUANTIZED_GEMM`` that supports the inputs (static
M <= 4 takes its own bucket; a dynamic M provably within [1, 4] takes the
smallest bucket that holds it). Everything else (prefill, an unbounded dynamic
M, unsupported group sizes, K or dtypes, CPU eager) uses inline dequant +
F.linear, never an error.

Keeping INT8 on a fused DP4A path lets mixed-precision recipes (e.g. INT8
edge-layer v_proj/down_proj + INT4 elsewhere) keep ALL decode linears fused
instead of materializing the full dequantized weight in HBM.

INT8 weights use the torchao ``IntxUnpackedToInt8Tensor`` subclass:
  qdata : [N, K]          int8 (one value per element, natural k order)
  scale : [N, K//gs]      bf16 (per-group, row-major)
  zero  : [N, K//gs]      int8 (per-group asymmetric zero point)

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the other formats)::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.quantize_op_dispatch._gemm_family_dispatch import (
    chunked_dequant_linear,
    quantized_linear,
)
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    IntxUnpackedToInt8Tensor,
)


def _unit_dq_mm_int8(x, qdata, scale, zero, group_size):
    """Dequant INT8 weights to input dtype and call F.linear.

    qdata [N, K] int8, scale/zero [N, K//gs]. Per-group asymmetric:
    w[n, k] = (qdata[n, k] - zero[n, k//gs]) * scale[n, k//gs].
    """
    N, K = qdata.shape
    n_groups = K // group_size
    dtype = x.dtype

    def dequant_linear_rows(i, j):
        rows = j - i
        q = qdata[i:j].to(dtype).reshape(rows, n_groups, group_size)
        s = scale[i:j].to(dtype).reshape(rows, n_groups, 1)
        z = zero[i:j].to(dtype).reshape(rows, n_groups, 1)
        return F.linear(x, ((q - z) * s).reshape(rows, K))

    return chunked_dequant_linear(x, N, dequant_linear_rows)


# ---------------------------------------------------------------------------
# IntxUnpackedToInt8Tensor F.linear dispatch
# ---------------------------------------------------------------------------

aten = torch.ops.aten
_implements_i8 = IntxUnpackedToInt8Tensor.implements
_implements_torch_function_i8 = IntxUnpackedToInt8Tensor.implements_torch_function


@_implements_i8([aten.linear.default])
@_implements_torch_function_i8([F.linear])
def _(func, types, args, kwargs):
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)

    if (
        weight.target_dtype is not torch.int8
        or weight.activation_quantization is not None
    ):
        return F.linear(input_tensor, weight.dequantize(), bias)

    from executorch.backends.cuda.triton.kernels.int8_quantized_gemm import (
        INT8_QUANTIZED_GEMM,
    )

    weight_args = (
        weight.qdata,
        weight.scale,
        weight.zero_point,
        weight.block_size[-1],
    )
    return quantized_linear(
        INT8_QUANTIZED_GEMM,
        input_tensor,
        weight_args,
        bias,
        lambda x_2d: _unit_dq_mm_int8(x_2d, *weight_args),
    )
