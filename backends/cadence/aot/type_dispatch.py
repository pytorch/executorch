# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch._ops import OpOverload
from torch.fx.node import Argument


@register_cadence_pass(CadencePassAttribute(opt_level=4))
class CompileTimeTypeDispatchPass(ExportPass):
    """
    Replaces generic ops with ops that have explicit types.
    """

    _BINARY_TYPE_DISPATCH_MAP: dict[tuple[torch.dtype, torch.dtype], str] = {
        (torch.int8, torch.int8): "asym8sxasym8s_asym8s",
        (torch.uint8, torch.uint8): "asym8uxasym8u_asym8u",
    }

    _UNARY_TYPE_DISPATCH_MAP: dict[torch.dtype, str] = {
        torch.int8: "asym8s_asym8s",
        torch.uint8: "asym8u_asym8u",
    }

    _BINARY_SUPPORTED_OPS: dict[OpOverload, str] = {
        exir_ops.edge.cadence.quantized_fully_connected.per_tensor: "quantized_fully_connected",
        exir_ops.edge.cadence.quantized_linear.per_tensor: "quantized_linear",
    }

    _SUPPORTED_UNARY_OPS: dict[OpOverload, str] = {
        exir_ops.edge.cadence.quantized_relu.per_tensor: "quantized_relu",
    }

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op in self._BINARY_SUPPORTED_OPS:
            # pyre-ignore[16]: None has no attribute `to_tensor`.
            input_dtype = args[0].to_tensor().dtype
            weight_dtype = args[1].to_tensor().dtype
            dtype_pair = (input_dtype, weight_dtype)

            if dtype_pair not in self._BINARY_TYPE_DISPATCH_MAP:
                raise RuntimeError(
                    f"Unsupported input types for {op}: {input_dtype} and {weight_dtype}"
                )

            base_op_name = self._BINARY_SUPPORTED_OPS[op]
            type_suffix = self._BINARY_TYPE_DISPATCH_MAP[dtype_pair]

            typed_op_name = f"{base_op_name}_{type_suffix}"
            typed_op = getattr(exir_ops.edge.cadence, typed_op_name).per_tensor

            return super().call_operator(typed_op, args, kwargs, meta)

        elif op in self._SUPPORTED_UNARY_OPS:
            input_dtype = args[0].to_tensor().dtype

            if input_dtype not in self._UNARY_TYPE_DISPATCH_MAP:
                raise RuntimeError(f"Unsupported input type for {op}: {input_dtype}")

            base_op_name = self._SUPPORTED_UNARY_OPS[op]
            type_suffix = self._UNARY_TYPE_DISPATCH_MAP[input_dtype]

            typed_op_name = f"{base_op_name}_{type_suffix}"
            typed_op = getattr(exir_ops.edge.cadence, typed_op_name).per_tensor

            return super().call_operator(typed_op, args, kwargs, meta)

        return super().call_operator(op, args, kwargs, meta)
