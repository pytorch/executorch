# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.dialects._ops import ops

from executorch.exir.pass_base import ExportPass, ProxyValue
from torch._subclasses.fake_tensor import FakeTensor
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.utils import _pytree as pytree

__all__ = [
    "quantized_decomposed_lib",
]


class ConstPropPass(ExportPass):
    """
    Performs constant folding and constant propagation.
    """

    def __init__(self, propogate_quant: bool = False) -> None:
        super().__init__()
        self.propogate_quant = propogate_quant

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        # pyre-ignore
        def is_const(arg) -> bool:
            if isinstance(arg, FakeTensor):
                return False
            if isinstance(
                arg,
                (
                    float,
                    int,
                    bool,
                    str,
                    torch.Tensor,
                    torch.device,
                    torch.dtype,
                    torch.layout,
                ),
            ):
                return True
            if isinstance(arg, (tuple, list)):
                return all(map(is_const, arg))
            if isinstance(arg, dict):
                return all(map(is_const, arg.values()))
            return False

        dequant_quant_ops = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ops.edge.quantized_decomposed.quantize_per_channel.default,
            ops.edge.quantized_decomposed.dequantize_per_channel.default,
        }
        op_is_q_dq = op in dequant_quant_ops
        # XNOR relationship, if propogate_quant is true only const prop quant ops,
        # if false propogate everything but quant ops
        if (
            (not op_is_q_dq and not self.propogate_quant)
            or (op_is_q_dq and self.propogate_quant)
        ) and is_const([args, kwargs]):
            guard = torch._C._DisableTorchDispatch()  # noqa
            try:
                args_data, kwargs_data = pytree.tree_map_only(
                    ProxyValue, lambda x: x.data, (args, kwargs)
                )
                result = op(*args_data, **kwargs_data)
            finally:
                del guard
            return result.to_tensor() if isinstance(result, ProxyValue) else result
        else:
            return super().call_operator(op, args, kwargs, meta)
