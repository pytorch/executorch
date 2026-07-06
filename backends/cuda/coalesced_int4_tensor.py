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

Metadata encoding: the SCALE is a per-group **uint8 code** with a
**per-256-super-block fp16 step** (``scale = code * step``); the ZERO now uses
the SAME per-256-super-block fp16 step (per-group uint8 code + per-256 fp16
step). group_size is 32, so a 256-weight super-block spans 8 groups and there
are ``K/256`` scale steps and ``K/256`` zero steps per row. Dequant is
unchanged: ``w = (q - zero) * scale`` with ``zero = min/scale``.

This mirrors GGUF Q4_K's per-super-block fp16 ``d`` for both the scale and the
zero: the finer per-256 step (vs the previous per-row step) is what lifts
whole-weight dequant SNR to ~45.89 dB (vs 45.15 dB for the per-row-zero-step
encoding, +0.74 dB). The scale/zero codes stay single coalesced uint8s (one
byte/group) so the decode kernel reads exactly one scale byte and one zero byte
per group — no bit-plane reconstruct — and both per-256 steps are loaded once
per super-block. Both steps MUST be fp16 (bf16 for the per-256 step costs ~0.05
dB on the scale; the per-256 zero step is +0.74 dB at fp16 vs +0.52 at bf16).

Layout difference from torchao ``Int4Tensor``:
    qdata      : packed int4 weight (N, K/2), nibble-packed (same as Int4Tensor)
    scale      : (N, n_groups) uint8 — per-group scale *codes*, coalesced
                 (transposed from torchao's (n_groups, N))
    scale_step : (N, K/256) fp16 — per-256-super-block scale step; the real
                 per-group scale is ``scale_code * scale_step[:, g // 8]``.
    zero_point : (N, n_groups) uint8 — per-group zero codes
    zero_step  : (N, K/256) fp16 — per-256-super-block zero step; the real
                 per-group zero is ``zero_code * zero_step[:, g // 8]``.

Bits-per-weight: 4.0 (qdata) + 8/32 (scale codes) + 16/256 (fp16 scale step) +
8/32 (uint8 zero codes) + 16/256 (fp16 zero step) = 4.625 bpw.

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

_CODE_MAX = 255  # uint8 code range [0, 255] (both scale and zero)
_SUPER_BLOCK = 256  # weights per super-block (GGUF Q4_K QK_K); scale step is per this


def _encode_uint8_per_super(
    x: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a (n_groups, N) non-negative tensor to per-super-block uint8 codes.

    Used for both the scale and the zero. A super-block is ``_SUPER_BLOCK``
    (256) weights = ``groups_per_super = 256 // group_size`` groups (8 for
    group_size=32). Returns ``(codes, step)`` where ``codes`` is
    ``(N, n_groups)`` uint8 (transposed to the coalesced layout) and ``step`` is
    ``(N, n_super)`` fp16 with ``n_super = n_groups // groups_per_super = K //
    256``, such that ``code * step[:, g // groups_per_super] ≈ x.t()``. The step
    is the per-256-super-block max / 255 rounded to fp16. Rounding uses the
    fp16-rounded step (what the kernel reads) so encode and decode agree.
    """
    xt = x.t().contiguous().float()  # (N, n_groups)
    N, n_groups = int(xt.shape[0]), int(xt.shape[1])
    groups_per_super = _SUPER_BLOCK // int(group_size)
    if groups_per_super < 1:
        raise ValueError(
            f"group_size={group_size} must be <= {_SUPER_BLOCK} for the per-256 "
            "scale step"
        )
    if n_groups % groups_per_super != 0:
        raise ValueError(
            f"n_groups={n_groups} must be a multiple of {groups_per_super} "
            f"(K must be a multiple of {_SUPER_BLOCK}) for group_size={group_size}"
        )
    n_super = n_groups // groups_per_super
    xb = xt.reshape(N, n_super, groups_per_super)  # (N, n_super, gps)
    block_max = xb.amax(dim=2, keepdim=True).clamp_min(1e-30)  # (N, n_super, 1)
    step = (block_max / _CODE_MAX).to(torch.float16)  # (N, n_super, 1) fp16
    step_f = step.float().clamp_min(1e-30)
    codes = torch.round(xb / step_f).clamp_(0, _CODE_MAX).to(torch.uint8)
    codes = codes.reshape(N, n_groups).contiguous()
    return codes, step.squeeze(2).contiguous()


def _unpack_nibble_qdata(qdata: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Unpack nibble-packed int4 qdata ``(N, K/2)`` -> ``(N, K)`` uint8 [0, 15]."""
    qu = qdata.to(torch.uint8)
    even = qu & 0xF
    odd = (qu >> 4) & 0xF
    return torch.stack([even, odd], dim=-1).reshape(N, K)


class CudaCoalescedInt4Tensor(TorchAOBaseTensor):
    """INT4 weight, uint8 scale + per-256 fp16 step, uint8 zero + per-256 fp16 step.

    ExecuTorch-internal; see the module docstring. Mirrors torchao
    ``Int4Tensor``'s data/attribute layout (so the common tensor utilities and
    serialization work) but owns the [n_groups, N] -> [N, n_groups] transpose and
    the uint8 re-encoding via :meth:`from_int4_tensor`.
    """

    tensor_data_names = [
        "qdata",
        "scale",
        "scale_step",
        "zero_point",
        "zero_step",
    ]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale"]
    optional_tensor_attribute_names = ["activation_dtype"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        scale_step: torch.Tensor,
        zero_point: torch.Tensor,
        zero_step: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = (
            activation_dtype if activation_dtype is not None else torch.bfloat16
        )
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        scale_step: torch.Tensor,
        zero_point: torch.Tensor,
        zero_step: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.scale_step = scale_step
        self.zero_point = zero_point
        self.zero_step = zero_step
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
        scale/zero_point as (n_groups, N) bf16. The CUDA decode kernel reads the
        (N, n_groups) uint8 scale/zero *codes* plus per-256-super-block
        (N, K/256) fp16 ``scale_step`` / ``zero_step`` (scale = scale_code *
        scale_step[:, g//8], zero = zero_code * zero_step[:, g//8]). The
        transpose + encode here is baked into the serialized weight constant so
        the exported decode graph has no per-step transpose/clone.
        """
        scale_codes, scale_step = _encode_uint8_per_super(t.scale, t.block_size[-1])
        zero_codes, zero_step = _encode_uint8_per_super(t.zero_point, t.block_size[-1])
        return cls(
            t.qdata,
            scale_codes,
            scale_step,
            zero_codes,
            zero_step,
            t.block_size,
            t.shape,
            t.act_pre_scale,
            t.activation_dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a dense tensor: ``w = (q - zero) * scale``.

        Reconstructs the per-group scale from the uint8 codes and the per-256
        fp16 step, and the per-group zero from the uint8 codes and the per-256
        fp16 step. Used as the numerical reference and for the tied lm_head /
        token embedding.
        """
        dtype = output_dtype if output_dtype is not None else torch.bfloat16
        N, K = int(self.shape[0]), int(self.shape[1])
        gs = self.block_size[-1]
        n_groups = K // gs
        n_super = int(self.scale_step.shape[1])
        groups_per_super = n_groups // n_super

        q = _unpack_nibble_qdata(self.qdata, N, K).to(torch.float32)
        scale_code = self.scale.to(torch.float32)  # (N, n_groups)
        scale_step = self.scale_step.float().repeat_interleave(groups_per_super, dim=1)
        scale = (scale_code * scale_step).repeat_interleave(gs, dim=1)  # (N, K)

        zero_code = self.zero_point.to(torch.float32)  # (N, n_groups)
        zero_step = self.zero_step.float().repeat_interleave(groups_per_super, dim=1)
        zero = (zero_code * zero_step).repeat_interleave(gs, dim=1)  # (N, K)
        return ((q - zero) * scale).to(dtype)


# Allow a model with CudaCoalescedInt4Tensor weights to be loaded with
# `weights_only=True` (mirrors torchao Int4Tensor).
torch.serialization.add_safe_globals([CudaCoalescedInt4Tensor])
