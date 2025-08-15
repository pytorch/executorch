# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Optional

import torch
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch._ops import OpOverload
from torch.fx.node import Argument


@dataclass
class OpConfig:
    """Configuration for type dispatch operations."""

    base_name: str
    input_arg_idx: int = 0
    weight_arg_idx: Optional[int] = None
    variant: str = "per_tensor"


@register_cadence_pass(CadencePassAttribute(opt_level=4))
class CompileTimeTypeDispatchPass(ExportPass):
    """
    Replaces generic ops with ops that have explicit types.
    """

    _TYPE_DISPATCH_MAP: dict[tuple[torch.dtype, ...], str] = {
        (torch.int8,): "asym8s_asym8s",
        (torch.uint8,): "asym8u_asym8u",
        (torch.int8, torch.int8): "asym8sxasym8s_asym8s",
        (torch.uint8, torch.uint8): "asym8uxasym8u_asym8u",
    }

    _SUPPORTED_OPS: dict[OpOverload, OpConfig] = {
        exir_ops.edge.cadence.quantized_fully_connected.per_tensor: OpConfig(
            "quantized_fully_connected", input_arg_idx=0, weight_arg_idx=1
        ),
        exir_ops.edge.cadence.quantized_linear.per_tensor: OpConfig(
            "quantized_linear", input_arg_idx=0, weight_arg_idx=1
        ),
        exir_ops.edge.cadence.quantized_matmul.default: OpConfig(
            "quantized_matmul", input_arg_idx=0, weight_arg_idx=2, variant="default"
        ),
        exir_ops.edge.cadence.quantized_relu.per_tensor: OpConfig(
            "quantized_relu", input_arg_idx=0
        ),
    }

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in self._SUPPORTED_OPS:
            return super().call_operator(op, args, kwargs, meta)

        config = self._SUPPORTED_OPS[op]

        # pyre-ignore[16]: None has no attribute `to_tensor`.
        input_dtype = args[config.input_arg_idx].to_tensor().dtype

        if config.weight_arg_idx is not None:
            weight_dtype = args[config.weight_arg_idx].to_tensor().dtype
            dtype_key = (input_dtype, weight_dtype)
        else:
            dtype_key = (input_dtype,)

        if dtype_key not in self._TYPE_DISPATCH_MAP:
            raise RuntimeError(f"Unsupported input types for {op}: {dtype_key}")

        type_suffix = self._TYPE_DISPATCH_MAP[dtype_key]
        typed_op_name = f"{config.base_name}_{type_suffix}"

        typed_op = getattr(
            getattr(exir_ops.edge.cadence, typed_op_name), config.variant
        )

        return super().call_operator(typed_op, args, kwargs, meta)
