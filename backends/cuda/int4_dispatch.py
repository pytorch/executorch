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

Import this module before using nn.Linear with Int4Tensor weights::

    import executorch.backends.cuda.int4_dispatch  # noqa: F401
"""

import torch
import torch.nn.functional as F
from torch.library import impl, Library
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

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
    return _dequant_matmul(self, qdata, scale, zero, group_size)


def _dequant_matmul(x, qdata, scale, zero, group_size):
    """Dequant INT4 weights to input dtype and call F.linear."""
    N, K_half = qdata.shape
    K = K_half * 2
    n_groups = K // group_size
    gs_half = group_size // 2
    dtype = x.dtype

    p = qdata.to(torch.uint8).reshape(N, n_groups, gs_half)
    low = (p & 0x0F).to(dtype)
    high = ((p >> 4) & 0x0F).to(dtype)
    data = torch.stack([low, high], dim=-1).reshape(N, n_groups, group_size)

    s = scale.to(dtype).t().unsqueeze(-1)
    z = zero.to(dtype).t().unsqueeze(-1)
    w_deq = ((data - z) * s).reshape(N, K)

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
        out = torch.ops.executorch_cuda.int4_plain_mm(x_2d, qdata, scale, zero, gs)
    else:
        out = _dequant_matmul(x_2d, qdata, scale, zero, gs)

    out = out.reshape(*orig_shape[:-1], -1)
    if bias is not None:
        out = out + bias
    return out
