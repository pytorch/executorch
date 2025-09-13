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
    type_dispatch_suffixes: dict[tuple[torch.dtype, ...], str]
    weight_arg_idx: Optional[int] = None
    variant: str = "per_tensor"


@register_cadence_pass(CadencePassAttribute(opt_level=4))
class CompileTimeTypeDispatchPass(ExportPass):
    """
    Replaces generic ops with ops that have explicit types.
    """

    _SUPPORTED_OPS: dict[OpOverload, OpConfig] = {
        exir_ops.edge.cadence.quantized_fully_connected.per_tensor: OpConfig(
            "quantized_fully_connected",
            type_dispatch_suffixes={
                (torch.int8, torch.int8): "asym8sxasym8s_asym8s",
                (torch.uint8, torch.uint8): "asym8uxasym8u_asym8u",
            },
            weight_arg_idx=1,
        ),
        exir_ops.edge.cadence.quantized_linear.per_tensor: OpConfig(
            "quantized_linear",
            type_dispatch_suffixes={
                (torch.int8, torch.int8): "asym8sxasym8s_asym8s",
                (torch.uint8, torch.uint8): "asym8uxasym8u_asym8u",
            },
            weight_arg_idx=1,
        ),
        exir_ops.edge.cadence.quantized_matmul.default: OpConfig(
            "quantized_matmul",
            type_dispatch_suffixes={
                (torch.int8, torch.int8): "asym8sxasym8s_asym8s",
                (torch.uint8, torch.uint8): "asym8uxasym8u_asym8u",
            },
            weight_arg_idx=2,
            variant="default",
        ),
        exir_ops.edge.cadence.quantized_conv_nchw.per_tensor: OpConfig(
            "quantized_conv_nchw",
            type_dispatch_suffixes={
                (torch.int8, torch.int8): "asym8sxsym8s_asym8s",
                (torch.uint8, torch.uint8): "asym8uxsym8u_asym8u",
            },
            weight_arg_idx=1,
        ),
        exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor: OpConfig(
            "quantized_conv_nhwc",
            type_dispatch_suffixes={
                (torch.int8, torch.int8): "asym8sxsym8s_asym8s",
                (torch.uint8, torch.uint8): "asym8uxsym8u_asym8u",
            },
            weight_arg_idx=1,
        ),
        exir_ops.edge.cadence.quantized_relu.per_tensor: OpConfig(
            "quantized_relu",
            type_dispatch_suffixes={
                (torch.int8,): "asym8s_asym8s",
                (torch.uint8,): "asym8u_asym8u",
            },
        ),
        exir_ops.edge.cadence.quantized_add.per_tensor: OpConfig(
            "quantized_add",
            type_dispatch_suffixes={
                (torch.int8, torch.int8): "asym8sxasym8s_asym8s",
                (torch.uint8, torch.uint8): "asym8uxasym8u_asym8u",
            },
            weight_arg_idx=3,
        ),
        exir_ops.edge.aten._softmax.default: OpConfig(
            "_softmax",
            type_dispatch_suffixes={
                (torch.float32,): "f32_f32",
            },
            variant="default",
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
        input_dtype = args[0].to_tensor().dtype

        if config.weight_arg_idx is not None:
            weight_dtype = args[config.weight_arg_idx].to_tensor().dtype
            dtype_key = (input_dtype, weight_dtype)
        else:
            dtype_key = (input_dtype,)

        if dtype_key not in config.type_dispatch_suffixes:
            raise RuntimeError(f"Unsupported input types for {op}: {dtype_key}")

        type_suffix = config.type_dispatch_suffixes[dtype_key]
        base_name = config.base_name

        typed_op_name = f"{base_name}_{type_suffix}"

        if op in [
            exir_ops.edge.cadence.quantized_conv_nchw.per_tensor,
            exir_ops.edge.cadence.quantized_conv_nhwc.per_tensor,
        ]:
            groups = args[6]
            input_channels = (
                args[0].to_tensor().shape[1]
                if op == exir_ops.edge.cadence.quantized_conv_nchw.per_tensor
                else args[0].to_tensor().shape[-1]
            )
            is_depthwise = groups == input_channels
            # pyre-ignore[16]: None has no attribute '__iter__'.
            is_dilated = any(d > 1 for d in args[5])
            is_1d = len(args[0].to_tensor().shape) == 3

            if is_depthwise:
                typed_op_name = f"{base_name}_depthwise_{type_suffix}"
            elif is_dilated:
                typed_op_name = f"{base_name}_dilated_{type_suffix}"
            elif is_1d and groups == 1:
                typed_op_name = (
                    f"quantized_conv1d_{base_name.split('_')[-1]}_{type_suffix}"
                )

        typed_op = getattr(
            getattr(exir_ops.edge.cadence, typed_op_name), config.variant
        )

        return super().call_operator(typed_op, args, kwargs, meta)
