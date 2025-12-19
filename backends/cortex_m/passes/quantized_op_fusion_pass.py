# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Sequence

import torch
from executorch.backends.cortex_m.passes.passes_utils import (
    quantize_multiplier_aot,
    SHIFT_INT8,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    CMSIS_SOFTMAX_SCALE,
    CMSIS_SOFTMAX_ZERO_POINT,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx.node import Argument
from torch.nn.modules.utils import _pair


class QuantizedOpFusionPass(ExportPass):
    """
    Generic ExportPass that:
    1. Replaces certain ops with cortex_m variants based on qualifiers.
    2. Fuses patterns: dequantize_per_tensor -> [binary_op] -> quantize_per_tensor
       into cortex_m.quantized_[op].default with AoT computed multipliers/shifts.


    Supports multiple binary operations with backward compatibility for add.
    """

    _SOFTMAX_INPUT_INTEGER_BITS = 5

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

    def _compute_softmax_params(self, input_scale: float) -> tuple[int, int, int]:
        """
        Convert the incoming per-tensor input scale into the CMSIS fixed-point
        parameters expected by `arm_softmax_s8`.

        1. Clamp the real multiplier to the Q31 range using the fixed number of
           input integer bits mandated by CMSIS.
        2. Feed that multiplier through `quantize_multiplier_aot` to get the
           (multiplier, shift) pair arm_softmax_s8 expects.
        3. Derive `diff_min`, the CMSIS threshold for early bailout when
           differences saturate, using the same multiplier/shift values.
        """
        real_multiplier = min(
            input_scale * (1 << (31 - self._SOFTMAX_INPUT_INTEGER_BITS)),
            float((1 << 31) - 1),
        )
        input_multiplier, input_shift = quantize_multiplier_aot(real_multiplier)
        diff_min_term = (
            ((1 << self._SOFTMAX_INPUT_INTEGER_BITS) - 1)
            * math.ldexp(1.0, 31 - self._SOFTMAX_INPUT_INTEGER_BITS)
            / math.ldexp(1.0, input_shift)
        )
        diff_min = -int(math.floor(diff_min_term))
        return int(input_multiplier), int(input_shift), diff_min

    def _get_softmax_replacement(self, args, meta):
        if (
            meta.data.get("input_qparams", {}) == {}
            or meta.data.get("output_qparams", {}) == {}
        ):
            return exir_ops.edge.aten._softmax.default, args

        input_qparams = meta["input_qparams"][0]
        output_qparams = meta["output_qparams"][0]

        half_to_float = args[2] if len(args) > 2 else False
        if half_to_float:
            return exir_ops.edge.aten._softmax.default, args

        input_multiplier, input_shift, diff_min = self._compute_softmax_params(
            float(input_qparams.scale)
        )

        output_scale_attr = getattr(output_qparams, "scale", None)
        output_zp_attr = getattr(output_qparams, "zp", None)
        if output_scale_attr is None or output_zp_attr is None:
            raise AssertionError("Softmax requires output quantization parameters.")

        output_scale_val = float(output_scale_attr)
        output_zp_val = int(output_zp_attr)
        if not math.isclose(
            output_scale_val, CMSIS_SOFTMAX_SCALE, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(
                "Softmax output scale must match CMSIS (1/256). "
                f"Got {output_scale_val}."
            )
        if output_zp_val != CMSIS_SOFTMAX_ZERO_POINT:
            raise AssertionError(
                "Softmax output zero-point must match CMSIS (-128). "
                f"Got {output_zp_val}."
            )

        new_args = (
            args[0],
            args[1],
            int(input_qparams.zp),
            output_zp_val,
            input_multiplier,
            input_shift,
            diff_min,
        )

        return exir_ops.edge.cortex_m.softmax.default, new_args

    def _unwrap_argument(self, arg: Argument) -> Argument:
        if isinstance(arg, ProxyValue):
            return arg.data
        return arg

    def _resolve_default(
        self, raw: Argument, default: Sequence[int] | None
    ) -> Argument:
        if raw is None:
            if default is None:
                raise RuntimeError("Expected default sequence for normalization")
            return default
        return raw

    def _coerce_to_int_list(self, raw: Argument) -> list[int]:
        if isinstance(raw, ProxyValue):
            raw = raw.data
        if isinstance(raw, torch.Tensor):
            return [int(v) for v in raw.flatten().tolist()]
        if isinstance(raw, (list, tuple, torch.Size)):
            return [int(v) for v in raw]
        if isinstance(raw, (int, bool)):
            return [int(raw)]

        try:
            first, second = _pair(raw)
        except TypeError as err:
            raise RuntimeError(
                f"Unsupported argument for pair normalization: {raw}"
            ) from err
        return [int(first), int(second)]

    def _normalize_int_pair(
        self, items: list[int], default: Sequence[int] | None
    ) -> list[int]:
        if not items:
            if default is None:
                raise RuntimeError("Cannot normalize empty sequence without default")
            items = [int(v) for v in default]

        if len(items) == 1:
            return [items[0], items[0]]
        if len(items) != 2:
            raise RuntimeError(
                f"Unsupported sequence length for pair normalization: {items}"
            )
        return [items[0], items[1]]

    def _to_int_pair(self, value: Argument, default: Sequence[int] | None) -> list[int]:
        raw = self._unwrap_argument(value)
        raw = self._resolve_default(raw, default)
        items = self._coerce_to_int_list(raw)
        return self._normalize_int_pair(items, default)

    def _to_bool(self, value: Argument, default: bool) -> bool:
        raw = self._unwrap_argument(value)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int):
            return bool(raw)
        if isinstance(raw, torch.Tensor):
            try:
                return bool(int(raw.item()))
            except Exception:
                return default
        return default

    def _get_max_pool2d_replacement(self, args, meta):
        input_qparams = meta["input_qparams"].get(0)
        cortex_m_meta = meta.data.get("custom", {}).get("cortex_m", {})
        if input_qparams is None or cortex_m_meta.get(
            "skip_quantized_max_pool2d", False
        ):
            return exir_ops.edge.aten.max_pool2d.default, args

        input_scale = float(input_qparams.scale)
        input_zero_point = int(input_qparams.zp)

        output_qparams = None
        if meta.data.get("output_qparams"):
            output_qparams = meta["output_qparams"].get(0)

        if output_qparams is not None:
            if getattr(output_qparams, "per_channel", False):
                return exir_ops.edge.aten.max_pool2d.default, args
            output_scale = float(output_qparams.scale)
            output_zero_point = int(output_qparams.zp)
            activation_min = int(output_qparams.qmin)
            activation_max = int(output_qparams.qmax)
            if abs(input_scale - output_scale) > 1e-6:
                return exir_ops.edge.aten.max_pool2d.default, args
            if input_zero_point != output_zero_point:
                return exir_ops.edge.aten.max_pool2d.default, args
        else:
            output_zero_point = input_zero_point
            activation_min = torch.iinfo(torch.int8).min
            activation_max = torch.iinfo(torch.int8).max

        kernel_size = self._to_int_pair(args[1], None)
        stride_arg = args[2] if len(args) > 2 else None
        stride = self._to_int_pair(stride_arg, kernel_size)
        padding_arg = args[3] if len(args) > 3 else None
        padding = self._to_int_pair(padding_arg, (0, 0))
        dilation_arg = args[4] if len(args) > 4 else None
        dilation = self._to_int_pair(dilation_arg, (1, 1))

        ceil_mode_arg = args[5] if len(args) > 5 else False
        ceil_mode = self._to_bool(ceil_mode_arg, False)

        if dilation != [1, 1] or ceil_mode:
            return exir_ops.edge.aten.max_pool2d.default, args

        quantized_op = getattr(exir_ops.edge.cortex_m, "quantized_max_pool2d", None)
        if quantized_op is None:
            return exir_ops.edge.aten.max_pool2d.default, args

        new_args = (
            args[0],
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            input_zero_point,
            output_zero_point,
            activation_min,
            activation_max,
        )

        return quantized_op.default, new_args

    def _get_minimum_replacement(self, args, meta):
        if args[0].data.dtype not in (torch.int8, torch.int32):
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
        kernel_size = self._to_int_pair(args[1], None)
        stride_arg = args[2] if len(args) > 2 else None
        stride = self._to_int_pair(stride_arg, kernel_size)
        padding_arg = args[3] if len(args) > 3 else None
        padding = self._to_int_pair(padding_arg, (0, 0))

        ceil_mode_arg = args[4] if len(args) > 4 else False
        ceil_mode = self._to_bool(ceil_mode_arg, False)
        count_include_pad_arg = args[5] if len(args) > 5 else True
        count_include_pad = self._to_bool(count_include_pad_arg, True)
        divisor_override = args[6] if len(args) > 6 else None
        divisor_override_val = self._unwrap_argument(divisor_override)

        if ceil_mode or count_include_pad or divisor_override_val is not None:
            return exir_ops.edge.aten.avg_pool2d.default, args

        args = (
            args[0],
            kernel_size,
            stride,
            padding,
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
            case exir_ops.edge.aten._softmax.default:
                op, args = self._get_softmax_replacement(args, meta)
            case exir_ops.edge.aten.max_pool2d.default:
                op, args = self._get_max_pool2d_replacement(args, meta)
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

        result = super().call_operator(op, args, {}, meta)
        return result
