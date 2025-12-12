# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from executorch.backends.cortex_m.passes.passes_utils import (
    quantize_multiplier_aot,
    SHIFT_INT8,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx.node import Argument


class QuantizedOpFusionPass(ExportPass):
    """
    Generic ExportPass that:
    1. Replaces certain ops with cortex_m variants based on qualifiers.
    2. Fuses patterns: dequantize_per_tensor -> [binary_op] -> quantize_per_tensor
       into cortex_m.quantized_[op].default with AoT computed multipliers/shifts.


    Supports multiple binary operations with backward compatibility for add.
    """

    def _get_add_replacement(self, args, meta):
        if (
            meta.data.get("input_qparams", {}) == {}
            or meta.data.get("output_qparams", {}) == {}
        ):
            return exir_ops.edge.aten.add.Tensor, args

        # Extract values
        scale1 = meta["input_qparams"][0].scale
        zero_point1 = meta["input_qparams"][0].zp
        scale2 = meta["input_qparams"][1].scale
        zero_point2 = meta["input_qparams"][1].zp
        output_scale = meta["output_qparams"][0].scale
        output_zero_point = meta["output_qparams"][0].zp

        # AoT COMPUTATION: Calculate multipliers and shifts
        max_scale_2x = 2 * max(scale1, scale2)

        input1_mult, input1_shift = quantize_multiplier_aot(scale1 / max_scale_2x)
        input2_mult, input2_shift = quantize_multiplier_aot(scale2 / max_scale_2x)
        output_mult, output_shift = quantize_multiplier_aot(
            max_scale_2x / (output_scale * (1 << SHIFT_INT8))
        )

        args = (
            args[0],
            zero_point1,
            input1_mult,
            input1_shift,
            args[1],
            zero_point2,
            input2_mult,
            input2_shift,
            output_zero_point,
            output_mult,
            output_shift,
        )

        return exir_ops.edge.cortex_m.quantized_add.default, args

    def _get_mul_replacement(self, args, meta):
        if (
            meta.data.get("input_qparams", {}) == {}
            or meta.data.get("output_qparams", {}) == {}
        ):
            return exir_ops.edge.aten.mul.Tensor, args

        # Extract values
        scale1 = meta["input_qparams"][0].scale
        zero_point1 = meta["input_qparams"][0].zp
        scale2 = meta["input_qparams"][1].scale
        zero_point2 = meta["input_qparams"][1].zp
        output_scale = meta["output_qparams"][0].scale
        output_zero_point = meta["output_qparams"][0].zp

        scale_factor = (scale1 * scale2) / output_scale
        output_mult, output_shift = quantize_multiplier_aot(scale_factor)

        args = (
            args[0],
            zero_point1,
            args[1],
            zero_point2,
            output_zero_point,
            output_mult,
            output_shift,
        )

        return exir_ops.edge.cortex_m.quantized_mul.default, args

    def _get_minimum_replacement(self, args, meta):
        if args[0].data.dtype != torch.int8:
            return exir_ops.edge.aten.minimum.default, args

        return exir_ops.edge.cortex_m.minimum.default, args

    def _get_maximum_replacement(self, args, meta):
        if args[0].data.dtype != torch.int8:
            return exir_ops.edge.aten.maximum.default, args

        return exir_ops.edge.cortex_m.maximum.default, args

    def _get_permute_replacement(self, args, meta):
        if args[0].data.dtype != torch.int8:
            return exir_ops.edge.aten.permute_copy.default, args

        rank = len(args[0].data.shape)
        perms = [p % rank for p in args[1]]
        args = (args[0], perms)
        return exir_ops.edge.cortex_m.transpose.default, args

    def _get_avg_pool2d_replacement(self, args, meta):
        if (
            meta.data.get("input_qparams", {}) == {}
            or meta.data.get("output_qparams", {}) == {}
        ):
            return exir_ops.edge.aten.avg_pool2d.default, args

        # Extract values
        scale = meta["input_qparams"][0].scale
        zero_point = meta["input_qparams"][0].zp

        output_mult, output_shift = quantize_multiplier_aot(scale)
        args = (
            *args[0:-2],
            zero_point,
            output_mult,
            output_shift,
        )

        return exir_ops.edge.cortex_m.quantized_avg_pool2d.default, args

    def call_operator(
        self,
        op: EdgeOpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:

        match op:
            case exir_ops.edge.aten.add.Tensor:
                op, args = self._get_add_replacement(args, meta)
            case exir_ops.edge.aten.mul.Tensor:
                op, args = self._get_mul_replacement(args, meta)
            case exir_ops.edge.aten.minimum.default:
                op, args = self._get_minimum_replacement(args, meta)
            case exir_ops.edge.aten.maximum.default:
                op, args = self._get_maximum_replacement(args, meta)
            case exir_ops.edge.aten.permute_copy.default:
                op, args = self._get_permute_replacement(args, meta)
            case exir_ops.edge.aten.avg_pool2d.default:
                op, args = self._get_avg_pool2d_replacement(args, meta)
            case _:
                pass

        return super().call_operator(op, args, {}, meta)
