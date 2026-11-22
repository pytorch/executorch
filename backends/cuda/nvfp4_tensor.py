# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CUDA NVFP4 weights: byte-packed E2M1 and N-major K16 E4M3 scales."""

import torch
from torch._subclasses.fake_tensor import is_fake
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
from torchao.utils import TorchAOBaseTensor


def _act_scale_tensor(act, device):
    """``act_per_tensor_scale`` as a checked float32 ``(1, 1)`` tensor."""
    if not isinstance(act, torch.Tensor):
        # Checked on the host, so meta and fake weights are checked too
        # and nothing is read back from the device.
        f32 = torch.finfo(torch.float32)
        if not f32.tiny <= act <= f32.max:
            raise ValueError(
                "CUDA NVFP4 act_per_tensor_scale must be finite and positive"
            )
        act = torch.tensor(act, dtype=torch.float32, device=device)
    elif act.numel() != 1 or act.dtype != torch.float32 or act.device != device:
        raise ValueError(
            "CUDA NVFP4 act_per_tensor_scale must be a scalar or a "
            "one-element float32 tensor on the weight's device"
        )
    # Fake and meta tensors carry no values to check.
    elif (
        act.device.type != "meta"
        and not is_fake(act)
        and not bool((torch.isfinite(act) & (act > 0)).all())
    ):
        raise ValueError("CUDA NVFP4 act_per_tensor_scale must be finite and positive")
    return act.reshape(1, 1).contiguous()


class CudaNVFP4Tensor(TorchAOBaseTensor):
    """Layout read by the CUDA kernels: ``qdata`` is ``(N, K / 2)`` uint8 with
    element ``2j`` in the low nibble of byte ``j`` (torchao order, unlike
    GGML's eight-lows-then-eight-highs); ``scale`` is ``(N, K / 16)`` E4M3
    bytes, N-major; ``per_tensor_scale`` is float32 of shape ``(1, 1)`` or
    ``(N,)`` (``fuse_along_output`` picks per-row scales when the inputs'
    scalars differ), so kernels handle both. The scalar is 2-D because
    Inductor inlines 1-D constants of <= 8 elements as fill kernels that run on
    every call. ``act_per_tensor_scale``, the optional static activation
    global scale, is likewise a float32 ``(1, 1)`` tensor: kernels read it from
    memory, so layers with different scales share compiled kernels and tuning.
    Its value keeps the ``ExportableNVFP4Tensor`` convention, the ModelOpt/GGUF
    ``input_scale`` ``amax / (6 * 448)`` (not its reciprocal); ``None`` means
    activations are scaled dynamically per row.

    ``unit_scale`` asserts two facts kernels may rely on: ``per_tensor_scale``
    is 1, so the global-scale multiply can be skipped, and every scale byte is
    below 0x7F, a non-negative finite E4M3 value, so each E2M1 x E4M3 product
    is finite with at most 6 significant bits and exact in BF16 without
    rounding. ``from_exportable`` sets it from the values; the constructor
    trusts the passed flag.

    Kernels assume every scale byte is a finite E4M3 value: ``from_exportable``
    rejects the NaN bytes 0x7F/0xFF (the GGUF converter already maps 0x7F to 0
    and rejects 0xFF).
    """

    tensor_data_names = ["qdata", "scale", "per_tensor_scale"]
    tensor_attribute_names = [
        "block_size",
        "orig_dtype",
        "unit_scale",
    ]
    optional_tensor_data_names = ["act_per_tensor_scale"]

    def __new__(
        cls,
        qdata,
        scale,
        per_tensor_scale,
        block_size=16,
        orig_dtype=torch.bfloat16,
        unit_scale=False,
        act_per_tensor_scale=None,
    ):
        if qdata.dtype != torch.uint8 or qdata.ndim != 2 or block_size != 16:
            raise ValueError(
                "CUDA NVFP4 requires 2D packed uint8 weights and block_size=16"
            )
        n, half_k = qdata.shape
        if half_k % 8:
            raise ValueError("CUDA NVFP4 K must be a multiple of 16")
        if scale.shape != (n, half_k // 8) or scale.dtype != torch.uint8:
            raise ValueError("CUDA NVFP4 scale shape must be (N, K/16) uint8")
        if not orig_dtype.is_floating_point:
            raise ValueError("NVFP4 output dtype must be floating point")
        if per_tensor_scale.numel() not in (1, n):
            raise ValueError(
                "CUDA NVFP4 per_tensor_scale must be scalar or have N values"
            )
        if per_tensor_scale.dtype != torch.float32:
            raise ValueError("CUDA NVFP4 per_tensor_scale must be float32")
        if scale.device != qdata.device or per_tensor_scale.device != qdata.device:
            raise ValueError("CUDA NVFP4 fields must be on one device")
        if act_per_tensor_scale is not None:
            act_per_tensor_scale = _act_scale_tensor(act_per_tensor_scale, qdata.device)
        qdata, scale = qdata.contiguous(), scale.contiguous()
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
        obj.unit_scale, obj.act_per_tensor_scale = unit_scale, act_per_tensor_scale
        return obj

    @classmethod
    def from_exportable(cls, w) -> "CudaNVFP4Tensor":
        """Repack a portable ``ExportableNVFP4Tensor`` (uint32 ``qdata``)."""
        # Infer once on materialized weights. Fake reconstruction receives the
        # serialized metadata through __new__, without reading tensor values.
        real = w.scale.device.type != "meta" and not is_fake(w.scale)
        if real and bool(((w.scale & 0x7F) == 0x7F).any().item()):
            raise ValueError("NVFP4 E4M3 scales must be finite (no 0x7F/0xFF bytes)")
        unit_scale = (
            real
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
            unit_scale,
            w.act_per_tensor_scale,
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
        act = first.act_per_tensor_scale
        if any(
            w.shape[1] != first.shape[1]
            or w.orig_dtype != first.orig_dtype
            or w.device != first.device
            or (w.act_per_tensor_scale is None) != (act is None)
            or (act is not None and not torch.equal(w.act_per_tensor_scale, act))
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
            all(w.unit_scale for w in weights),
            act,
        )

    def dequantize(self, output_dtype=None):
        values = f4_unpacked_to_f32(unpack_uint4(self.qdata))
        s = self.scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        pts = self.per_tensor_scale.reshape(-1, 1)
        return (values * (s * pts)).to(output_dtype or self.orig_dtype)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device, dtype = kwargs["device"], kwargs["dtype"]
        act = self.act_per_tensor_scale
        return type(self)(
            self.qdata.to(device),
            self.scale.to(device),
            self.per_tensor_scale.to(device),
            self.block_size,
            dtype,
            self.unit_scale,
            None if act is None else act.to(device),
        )


torch.serialization.add_safe_globals([CudaNVFP4Tensor])
