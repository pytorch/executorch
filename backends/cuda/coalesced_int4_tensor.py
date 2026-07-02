# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ExecuTorch-internal INT4 tensor for the CUDA W4A8 dp4a decode kernel.

``CudaCoalescedInt4Tensor`` is an ExecuTorch-internal tensor subclass. It is
**NOT** torchao's ``Int4Tensor`` and is intentionally not a subclass of it, so
torchao's ``Int4Tensor`` F.linear handlers never match it via the method
resolution order. The CUDA decode/prefill dispatch (``int4_dispatch.py``) is
selected by *type* — it is registered on this class only — so stock
``Int4Tensor`` weights keep falling back to torchao's default (mslk/tinygemm)
path.

Layout difference from torchao ``Int4Tensor``:
    qdata      : packed int4 weight (N, K/2), nibble-packed (same as Int4Tensor)
    scale      : (N, n_groups) uint8 — per-group scale *codes*, coalesced
                 (transposed from torchao's (n_groups, N))
    zero_point : (N, n_groups) uint8 — per-group zero *codes*, coalesced
    steps      : (N, 2) bf16 — per-row super-scales (scale_step, zero_step);
                 the real per-group values are ``code * step``. This compacts
                 the metadata from bf16 scale + bf16 zero (4 B/group, 5.0 bpw)
                 to uint8 scale + uint8 zero + a tiny per-row step (2 B/group,
                 4.5 bpw) at ~baseline accuracy (Q4_K group scales fit an
                 8-bit per-row-normalized code; measured dequant SNR 48 dB,
                 identical to the bf16 metadata it replaces).

The coalesced [N, n_groups] layout is exactly what the W4A8 dp4a matvec kernel
(``executorch_cuda::int4_plain_mm`` / ``int4_plain_mm.cuh``) reads row-for-row
with qdata, so the exported decode graph carries no per-step transpose. The
transpose (and the uint8 re-encoding) is owned by :meth:`from_int4_tensor` so it
is baked into the serialized weight constant once at pack time.
"""

from typing import List, Optional, Tuple

import torch
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "CudaCoalescedInt4Tensor",
]


def _encode_uint8_per_row(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a (n_groups, N) non-negative tensor to per-row uint8 codes.

    Returns ``(codes, step)`` where ``codes`` is ``(N, n_groups)`` uint8
    (already transposed to the coalesced layout) and ``step`` is ``(N,)`` bf16,
    such that ``code * step ≈ x.t()``. The step is the per-row max / 255 rounded
    to bf16, so the largest group in each row maps to ~255 and the 8-bit code
    spans the row's dynamic range. Q4_K scales/zeros are non-negative, so an
    unsigned code is exact at the endpoints and ~baseline accuracy elsewhere.
    """
    xt = x.t().contiguous().float()  # (N, n_groups)
    row_max = xt.amax(dim=1, keepdim=True).clamp_min(1e-30)  # (N, 1)
    step = (row_max / 255.0).to(torch.bfloat16)  # (N, 1) bf16
    step_f = step.float().clamp_min(1e-30)
    codes = torch.round(xt / step_f).clamp_(0, 255).to(torch.uint8)
    return codes.contiguous(), step.squeeze(1).contiguous()


class CudaCoalescedInt4Tensor(TorchAOBaseTensor):
    """INT4 weight with scale/zero_point in the coalesced [N, n_groups] layout.

    ExecuTorch-internal; see the module docstring. Mirrors torchao
    ``Int4Tensor``'s data/attribute layout (so the common tensor utilities and
    serialization work) but owns the [n_groups, N] -> [N, n_groups] transpose
    of scale/zero_point via :meth:`from_int4_tensor`.
    """

    tensor_data_names = ["qdata", "scale", "zero_point", "steps"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale"]
    optional_tensor_attribute_names = ["activation_dtype"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        steps: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = steps.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        steps: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.steps = steps
        self.block_size = block_size
        self.activation_dtype = (
            activation_dtype if activation_dtype is not None else torch.bfloat16
        )
        self.act_pre_scale = act_pre_scale

    def _quantization_type(self):
        s = f"shape={self.shape}, block_size={self.block_size}, device={self.device}, activation_dtype={self.activation_dtype}"
        if self.act_pre_scale is not None:
            s += f", act_pre_scale.shape={self.act_pre_scale.shape}"
        return s

    @classmethod
    def from_int4_tensor(cls, t: Int4Tensor) -> "CudaCoalescedInt4Tensor":
        """Build a coalesced tensor from a torchao ``Int4Tensor``.

        Owns the transpose AND the uint8 re-encoding: torchao stores
        scale/zero_point as (n_groups, N) bf16; the CUDA decode kernel reads
        (N, n_groups) uint8 *codes* plus a per-row (N, 2) bf16 ``steps``
        super-scale (scale = code*scale_step, zero = code*zero_step). The
        transpose + encode here is baked into the serialized weight constant so
        the exported decode graph has no per-step transpose/clone, and the
        constant is 2 B/group instead of 4 B/group (5.0 -> 4.5 bpw).
        """
        scale_codes, scale_step = _encode_uint8_per_row(t.scale)
        zero_codes, zero_step = _encode_uint8_per_row(t.zero_point)
        steps = torch.stack([scale_step, zero_step], dim=1).contiguous()  # (N, 2)
        return cls(
            t.qdata,
            scale_codes,
            zero_codes,
            steps,
            t.block_size,
            t.shape,
            t.act_pre_scale,
            t.activation_dtype,
        )


# Allow a model with CudaCoalescedInt4Tensor weights to be loaded with
# `weights_only=True` (mirrors torchao Int4Tensor).
torch.serialization.add_safe_globals([CudaCoalescedInt4Tensor])
