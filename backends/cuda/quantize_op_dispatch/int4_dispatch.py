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
  - On SM90+, prefill uses a pure-Triton Q4_K -> FP8 -> BF16-output linear.
  - Older GPUs use inline dequant + F.linear, compiled into a Triton dequant
    and BF16 matmul.

Dispatch strategy (determines what gets captured in the export graph):
  Decode (M<=4): Custom op ``executorch_cuda::int4_plain_mm``
  Prefill (M>4): FP8 Triton linear on SM90+, existing BF16 path otherwise

Importing the parent ``quantize_op_dispatch`` package registers this dispatch
override (along with the INT8 one) before using nn.Linear with
CudaCoalescedInt4Tensor weights::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.backends.cuda.quantize_op_dispatch._library import lib as _lib
from executorch.backends.cuda.quantize_op_dispatch.q4k_dequant import dequant_matmul
from executorch.backends.cuda.target_arch import cuda_targets_are_sm90_or_newer
from executorch.backends.cuda.triton.kernels.q4k_fp8_linear import q4k_fp8_linear
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
    # fp16 zero_point_step. dequant_matmul reconstructs scale =
    # code*scale_step[g//8], zero = code*zero_point_step[g//8].
    return dequant_matmul(
        self, qdata, scale, scale_step, zero, zero_point_step, group_size
    )


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
        # CUDA export traces with CPU example tensors, then the CUDA backend
        # lowers the captured custom op. Treat tracing as a CUDA-target case;
        # requiring ``x_2d.is_cuda`` here silently falls back to BF16 dequant
        # during export even when the target GPU is SM90+.
        cuda_target = x_2d.is_cuda or torch.compiler.is_compiling()
        if (
            cuda_targets_are_sm90_or_newer()
            and cuda_target
            and x_2d.dtype == torch.bfloat16
            and x_2d.is_contiguous()
            and gs == 32
            and x_2d.shape[1] % 256 == 0
        ):
            out = q4k_fp8_linear(
                x_2d,
                qdata,
                scale,
                scale_step,
                zero,
                zero_point_step,
                gs,
            )
        else:
            out = dequant_matmul(
                x_2d, qdata, scale, scale_step, zero, zero_point_step, gs
            )

    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out
