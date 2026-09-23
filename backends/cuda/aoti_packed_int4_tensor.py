# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nibble-packed W4/BF16 weights for the AOTI Triton matmul path."""

import torch
import torch.nn as nn
from executorch.backends.cuda.triton.kernels.int4_matmul import (
    int4_matmul,
    int4_matvec_bf16,
)
from torchao.utils import TorchAOBaseTensor


class AotiPackedInt4Tensor(TorchAOBaseTensor):
    """Symmetric groupwise INT4 weight consumed by AOTI Triton kernels.

    Linears use ``triton::int4_matmul`` by default; a fixed-shape caller can
    select ``triton::int4_matvec_bf16``.
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = ["group_size", "orig_dtype", "use_matvec"]

    def __new__(cls, qdata, scale, group_size, orig_dtype, use_matvec=False):
        shape = (qdata.shape[0], qdata.shape[1] * 2)
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=orig_dtype,
            device=qdata.device,
            requires_grad=False,
        )
        self.qdata = qdata
        self.scale = scale
        self.group_size = group_size
        self.orig_dtype = orig_dtype
        self.use_matvec = use_matvec
        return self

    @classmethod
    def from_intx_unpacked_to_int8_tensor(
        cls, weight: torch.Tensor, use_matvec: bool = False
    ) -> "AotiPackedInt4Tensor":
        from torchao.quantization import IntxUnpackedToInt8Tensor

        if not isinstance(weight, IntxUnpackedToInt8Tensor):
            raise TypeError(
                "AotiPackedInt4Tensor requires IntxUnpackedToInt8Tensor, got "
                f"{type(weight).__name__}"
            )
        if weight.target_dtype != torch.int4:
            raise ValueError("AotiPackedInt4Tensor requires target_dtype=torch.int4")
        if weight.activation_quantization is not None:
            raise ValueError("AotiPackedInt4Tensor only supports weight-only INT4")
        if weight.dtype != torch.bfloat16:
            raise ValueError("AotiPackedInt4Tensor requires BF16 activations")
        if weight.qdata.ndim != 2 or weight.block_size[0] != 1:
            raise ValueError("AotiPackedInt4Tensor requires 2D groupwise weights")
        if torch.count_nonzero(weight.zero_point).item() != 0:
            raise ValueError("AotiPackedInt4Tensor requires symmetric zero points")
        group_size = int(weight.block_size[-1])
        rows, columns = weight.qdata.shape
        if columns % 2 != 0 or columns % group_size != 0:
            raise ValueError(
                "AotiPackedInt4Tensor requires the input dimension to be "
                "divisible by two and the group size"
            )
        if weight.scale.shape != (rows, columns // group_size):
            raise ValueError("AotiPackedInt4Tensor received incompatible scales")

        qdata = (weight.qdata + 8).to(torch.uint8)
        qdata = (qdata[..., ::2] | (qdata[..., 1::2] << 4)).to(torch.int8)
        return cls(
            qdata.contiguous(),
            weight.scale.contiguous(),
            group_size,
            weight.dtype,
            use_matvec,
        )

    def dequantize(self, output_dtype=None):
        dtype = output_dtype or self.orig_dtype
        packed = self.qdata.to(torch.uint8)
        low = (packed & 0x0F).to(torch.float32) - 8.0
        high = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0
        values = torch.stack([low, high], dim=-1).reshape(self.shape)
        scale = self.scale.float().repeat_interleave(self.group_size, dim=-1)
        return (values * scale).to(dtype)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        dtype = kwargs.pop("dtype")
        if dtype != torch.bfloat16:
            raise ValueError("AotiPackedInt4Tensor requires BF16 activations")
        return AotiPackedInt4Tensor(
            self.qdata.to(device),
            self.scale.to(device=device, dtype=dtype),
            self.group_size,
            dtype,
            self.use_matvec,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


implements = AotiPackedInt4Tensor.implements


@implements([torch.ops.aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    input_shape = input_tensor.shape
    input_2d = input_tensor.reshape(-1, input_shape[-1])
    rows = input_2d.shape[0]
    if isinstance(rows, int) and rows == 1 and weight.use_matvec:
        output = int4_matvec_bf16(
            input_2d,
            weight.qdata,
            weight.scale,
            weight.group_size,
        )
    else:
        output = int4_matmul(
            input_2d,
            weight.qdata,
            weight.scale,
            weight.group_size,
        )
    output = output.reshape(*input_shape[:-1], weight.shape[0])
    return output if bias is None else output + bias


@implements([torch.ops.aten.detach.default, torch.ops.aten.alias.default])
def _(func, types, args, kwargs):
    return args[0]


@implements([torch.ops.aten.t.default])
def _(func, types, args, kwargs):
    return args[0].dequantize().t()


@implements([torch.ops.aten._to_copy.default])
def _(func, types, args, kwargs):
    return args[0].to(**kwargs)


def pack_int4_weights_for_aoti(module: nn.Module, *, use_matvec: bool = False) -> int:
    """Pack symmetric TorchAO INT4 linear weights in ``module`` in place."""
    from torchao.quantization import IntxUnpackedToInt8Tensor

    packed = 0
    for child in module.modules():
        if not isinstance(child, nn.Linear):
            continue
        weight = child.weight
        if not isinstance(weight, IntxUnpackedToInt8Tensor):
            continue
        if weight.target_dtype != torch.int4:
            continue
        child.weight = nn.Parameter(
            AotiPackedInt4Tensor.from_intx_unpacked_to_int8_tensor(
                weight,
                use_matvec=use_matvec,
            ),
            requires_grad=False,
        )
        packed += 1
    return packed


torch.serialization.add_safe_globals([AotiPackedInt4Tensor])
