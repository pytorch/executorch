# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, cast, NamedTuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops

exir_ops = cast(Any, exir_ops)
from executorch.backends.arm.constants import PER_CHANNEL_QDQ_OPS, PER_TENSOR_QDQ_OPS
from torch import Tensor


class QuantArgs(NamedTuple):
    scale: list[float] | float
    zp: list[int] | int
    qmin: int
    qmax: int
    dtype: torch.dtype
    axis: int = 0
    per_channel: bool = False

    def quantize_value(self, x: torch.Tensor | float) -> Tensor:
        """Quantizes the input tensor or value to a quantized tensor. If the input is
        not a tensor, it is converted to a tensor first. If self.per_channel is True,
        the quantization is done per channel, otherwise it is done per tensor.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor([x])
        x = x.to(torch.float32)
        if self.per_channel:
            q_op = exir_ops.edge.quantized_decomposed.quantize_per_channel.default
            args = (
                x,
                torch.tensor(self.scale),
                torch.tensor(self.zp),
                self.axis,
                self.qmin,
                self.qmax,
                self.dtype,
            )
        else:
            q_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            args = (x, self.scale, self.zp, self.qmin, self.qmax, self.dtype)  # type: ignore[assignment]
        return q_op(*args)

    def dequantize_value(self, qx: torch.Tensor) -> torch.Tensor:
        """Dequantizes the input tensor or value to a dequantized tensor  If the input
        is not a tensor, it is converted to a tensor first. If self.per_channel is True,
        the dequantization is done per channel, otherwise it is done per tensor.
        """
        if self.per_channel:
            dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
            args = (
                qx,
                torch.tensor(self.scale),
                torch.tensor(self.zp),
                self.axis,
                self.qmin,
                self.qmax,
                self.dtype,
            )
        else:
            dq_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            args = (qx, self.scale, self.zp, self.qmin, self.qmax, self.dtype)  # type: ignore[assignment]
        return dq_op(*args)

    @classmethod
    def from_operator(cls, op, args):
        if op in PER_TENSOR_QDQ_OPS:
            return cls(
                scale=cast(float, args[1]),
                zp=cast(int, args[2]),
                qmin=cast(int, args[3]),
                qmax=cast(int, args[4]),
                dtype=cast(torch.dtype, args[5]),
                axis=0,
                per_channel=False,
            )
        elif op in PER_CHANNEL_QDQ_OPS:
            return cls(
                scale=cast(list[float], args[1].tolist()),
                zp=cast(list[int], args[2].tolist()),
                axis=cast(int, args[3]),
                qmin=cast(int, args[4]),
                qmax=cast(int, args[5]),
                dtype=cast(torch.dtype, args[6]),
                per_channel=True,
            )
        else:
            # We're only handling per tensor and per channel quantization
            raise NotImplementedError(f"Unsupported quantization operation: {op}")

    def get_scale_per_tensor(self) -> float:
        if not isinstance(self.scale, float):
            raise TypeError(
                f"Expected scale {self.scale} to be a float but found scale of "
                f"type {type(self.scale)}"
            )
        return self.scale

    def get_zp_per_tensor(self) -> int:
        if not isinstance(self.zp, int):
            raise TypeError(
                f"Expected zero point {self.zp} to be an int but found zp of "
                f"type {type(self.zp)}"
            )
        return self.zp

    def get_scale_per_channel(self) -> list[float]:
        if not isinstance(self.scale, list):
            raise TypeError(
                f"Expected scale {self.scale} to be a list but found scale of "
                f"type {type(self.scale)}"
            )
        return self.scale

    def get_zp_per_channel(self) -> list[int]:
        if not isinstance(self.zp, list):
            raise TypeError(
                f"Expected zero point {self.zp} to be a list but found zp of "
                f"type {type(self.zp)}"
            )
        return self.zp
