# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CUDA NVFP4 weights: byte-packed E2M1 and N-major K16 E4M3 scales."""

import torch
from torch._subclasses.fake_tensor import is_fake
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
from torchao.utils import TorchAOBaseTensor


class CudaNVFP4Tensor(TorchAOBaseTensor):
    """Layout read by the CUDA kernels: ``qdata`` is ``(N, K / 2)`` uint8 with
    element ``2j`` in the low nibble of byte ``j`` (torchao order, unlike
    GGML's eight-lows-then-eight-highs); ``scale`` is ``(N, K / 16)`` E4M3
    bytes, N-major; ``per_tensor_scale`` is float32 of shape ``(1, 1)`` or
    ``(N,)`` (``fuse_along_output`` picks per-row scales when the inputs'
    scalars differ), so kernels handle both. The scalar is 2-D because
    Inductor inlines 1-D constants of <= 8 elements as fill kernels that run on
    every call. ``act_per_tensor_scale`` keeps
    its ``ExportableNVFP4Tensor`` meaning.

    ``unit_scale`` asserts two facts kernels may rely on: ``per_tensor_scale``
    is 1, so the global-scale multiply can be skipped, and every scale byte is
    below 0x7F, a non-negative finite E4M3 value, so each E2M1 x E4M3 product
    is finite with at most 6 significant bits and exact in BF16 without
    rounding. ``from_exportable`` sets it from the values; the constructor
    trusts the passed flag.
    """

    tensor_data_names = ["qdata", "scale", "per_tensor_scale"]
    tensor_attribute_names = [
        "block_size",
        "orig_dtype",
        "act_per_tensor_scale",
        "unit_scale",
    ]

    def __new__(
        cls,
        qdata,
        scale,
        per_tensor_scale,
        block_size=16,
        orig_dtype=torch.bfloat16,
        act_per_tensor_scale=None,
        unit_scale=False,
    ):
        if qdata.dtype != torch.uint8 or qdata.ndim != 2 or block_size != 16:
            raise ValueError(
                "CUDA NVFP4 requires 2D packed uint8 weights and block_size=16"
            )
        n, half_k = qdata.shape
        if half_k % 8 or scale.shape != (n, half_k // 8) or scale.dtype != torch.uint8:
            raise ValueError("CUDA NVFP4 scale shape must be (N, K/16) uint8")
        if per_tensor_scale.numel() not in (1, n):
            raise ValueError(
                "CUDA NVFP4 per_tensor_scale must be scalar or have N values"
            )
        if per_tensor_scale.dtype != torch.float32:
            raise ValueError("CUDA NVFP4 per_tensor_scale must be float32")
        if scale.device != qdata.device or per_tensor_scale.device != qdata.device:
            raise ValueError("CUDA NVFP4 fields must be on one device")
        qdata, scale = qdata.contiguous(), scale.contiguous()
        # A one-element tensor remains a device pointer during AOTI lowering.
        # A CPU zero-dimensional tensor may be folded into an IR scalar,
        # which cannot be passed to a Triton pointer argument.
        per_tensor_scale = per_tensor_scale.reshape(
            (1, 1) if per_tensor_scale.numel() == 1 else (-1,)
        ).contiguous()
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            (n, half_k * 2),
            dtype=orig_dtype,
            device=qdata.device,
            requires_grad=False,
        )
        obj.qdata, obj.scale, obj.per_tensor_scale = qdata, scale, per_tensor_scale
        obj.block_size, obj.orig_dtype = block_size, orig_dtype
        obj.act_per_tensor_scale = (
            None if act_per_tensor_scale is None else float(act_per_tensor_scale)
        )
        obj.unit_scale = unit_scale
        return obj

    @classmethod
    def from_exportable(cls, w) -> "CudaNVFP4Tensor":
        """Repack a portable ``ExportableNVFP4Tensor`` (uint32 ``qdata``)."""
        # Infer once on materialized weights. Fake reconstruction receives the
        # serialized metadata through __new__, without reading tensor values.
        unit_scale = (
            w.scale.device.type != "meta"
            and not is_fake(w.scale)
            and not is_fake(w.per_tensor_scale)
            and w.scale.numel() > 0
            and bool(torch.all(w.per_tensor_scale == 1).item())
            and bool(torch.all(w.scale < 127).item())
        )
        return cls(
            w.qdata.contiguous().view(torch.uint8),
            w.scale,
            w.per_tensor_scale,
            w.block_size,
            w.orig_dtype,
            w.act_per_tensor_scale,
            unit_scale,
        )

    @classmethod
    def fuse_along_output(cls, weights):
        # Fusing projections (e.g. gate/up) requires concatenating their packed
        # data and block scales together. Keep differing global scales per output
        # row so the fused weights need no requantization.
        weights = list(weights)
        if not weights:
            raise ValueError("CUDA NVFP4 fusion requires at least one weight")
        if any(not isinstance(w, cls) for w in weights):
            raise TypeError("CUDA NVFP4 fusion requires NVFP4 weights")
        first = weights[0]
        if any(
            w.shape[1] != first.shape[1]
            or w.orig_dtype != first.orig_dtype
            or w.device != first.device
            or w.act_per_tensor_scale != first.act_per_tensor_scale
            for w in weights[1:]
        ):
            raise ValueError(
                "CUDA NVFP4 fusion requires matching K, dtype, device and activation scale"
            )
        if len(weights) == 1:
            return first
        pts = first.per_tensor_scale
        if not all(
            w.per_tensor_scale.numel() == 1 and torch.equal(w.per_tensor_scale, pts)
            for w in weights
        ):
            pts = torch.cat(
                [w.per_tensor_scale.reshape(-1).expand(w.shape[0]) for w in weights]
            )
        return cls(
            torch.cat([w.qdata for w in weights]),
            torch.cat([w.scale for w in weights]),
            pts,
            first.block_size,
            first.orig_dtype,
            first.act_per_tensor_scale,
            all(w.unit_scale for w in weights),
        )

    def dequantize(self, output_dtype=None):
        values = f4_unpacked_to_f32(unpack_uint4(self.qdata))
        s = self.scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        pts = self.per_tensor_scale.reshape(-1, 1)
        return (values * (s * pts)).to(output_dtype or self.orig_dtype)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device, dtype = kwargs["device"], kwargs["dtype"]
        if not dtype.is_floating_point:
            raise ValueError("NVFP4 output dtype must be floating point")
        return type(self)(
            self.qdata.to(device),
            self.scale.to(device),
            self.per_tensor_scale.to(device),
            self.block_size,
            dtype,
            self.act_per_tensor_scale,
            self.unit_scale,
        )


torch.serialization.add_safe_globals([CudaNVFP4Tensor])
