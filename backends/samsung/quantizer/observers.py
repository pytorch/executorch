# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver


_EPS = torch.finfo(torch.float32).eps


class _FP16RepresentableZeroPointMixin:
    """
    Snap zero_point to the nearest fp16-representable integer: fp16 stores
    integers exactly only within [-2048, 2048] and even multiples of
    increasing powers of two beyond that. The snapped value is clamped to
    the nearest representable integer inside [quant_min, quant_max], since
    snapping may round a boundary value (e.g. 32767) past the range.
    """

    # largest integer exactly representable in fp16
    _FP16_INT_MAX = 65504

    @classmethod
    def _fp16_floor(cls, v):
        # largest fp16-representable integer <= v
        if v >= cls._FP16_INT_MAX:
            return cls._FP16_INT_MAX
        if v > 2048:
            spacing = 1 << (v.bit_length() - 11)
            return (v // spacing) * spacing
        if v >= -2048:
            return v
        return -cls._fp16_ceil(-v)

    @classmethod
    def _fp16_ceil(cls, v):
        # smallest fp16-representable integer >= v
        if v >= cls._FP16_INT_MAX:
            return cls._FP16_INT_MAX
        if v > 2048:
            spacing = 1 << (v.bit_length() - 11)
            return -((-v) // spacing) * spacing
        if v >= -2048:
            return v
        return -cls._fp16_floor(-v)

    def _calculate_qparams(self, min_val, max_val):
        scale, zero_point = super()._calculate_qparams(min_val, max_val)
        if self.qscheme == torch.per_channel_affine_float_qparams:
            # zero_point is fractional by design (xq = round(x / scale) + zp)
            return scale, zero_point
        zp = zero_point.to(torch.float16)
        # zero_point beyond fp16 max (65504) overflows to inf during the cast
        zp = torch.nan_to_num(
            zp,
            nan=0.0,
            posinf=float(self._FP16_INT_MAX),
            neginf=float(-self._FP16_INT_MAX),
        )
        zp = zp.to(zero_point.dtype)
        if self.quant_min is not None:
            zp = torch.clamp(zp, min=self._fp16_ceil(self.quant_min))
        if self.quant_max is not None:
            zp = torch.clamp(zp, max=self._fp16_floor(self.quant_max))
        if self.quant_min is not None and self.quant_max is not None:
            # A zero point pinned at quant_min/quant_max leaves no headroom on
            # that side of the grid (its edge dequantizes to 0), and no scale
            # can cover a range extending past it. Step zp inward to the next
            # fp16-representable integer where the observed range requires it.
            need_min_side = (zp == self.quant_min) & (min_val < 0)
            need_max_side = (zp == self.quant_max) & (max_val > 0)
            zp = torch.where(need_min_side, self._fp16_ceil(self.quant_min + 1), zp)
            zp = torch.where(need_max_side, self._fp16_floor(self.quant_max - 1), zp)
        scale = self._cover_range_after_snap(scale, min_val, max_val, zero_point, zp)
        return scale, zp

    def _cover_range_after_snap(self, scale, min_val, max_val, zp_orig, zp):
        # Snapping shifts the grid by (zp - zp_orig) * scale, which can clip the
        # range edges. Where zp changed, take the minimal scale that still covers
        # [min_val, max_val] for the snapped zp (half-step margin for rounding).
        if (
            self.quant_min is None
            or self.quant_max is None
            or self.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric)
        ):
            return scale
        changed = zp != zp_orig
        if not torch.any(changed):
            return scale
        min_neg = torch.minimum(min_val, torch.zeros_like(min_val))
        max_pos = torch.maximum(max_val, torch.zeros_like(max_val))
        covering = torch.maximum(
            max_pos / (self.quant_max - 0.5 - zp),
            -min_neg / (zp - self.quant_min - 0.5),
        )
        covering = torch.maximum(covering, self.eps)
        return torch.where(changed, covering, scale)


class MinMaxObserverWithFP16ZeroPoint(_FP16RepresentableZeroPointMixin, MinMaxObserver):
    pass


class PerChannelClippingObserver(PerChannelMinMaxObserver):
    """
    Clipping min/max range as MSE of weights, for SpinQuant + A16W4 only, because
    the evaluation of SpinQuant requires max-clipping with MSE and GPTQ.

    The range is searched once, on the first tensor it sees, and forward returns
    that tensor fake-quantized with it rather than unchanged.
    """

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=_EPS,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        self.steps = 200
        self.clipping_lower_limit = 0.1  # lower_limit of clipping range
        self.calibrated = False

    def reset_min_max_vals(self):
        super().reset_min_max_vals()
        self.calibrated = False

    def mse_per_ch(self, pred, expect):
        mse = (pred - expect).abs().pow(2)
        per_ch_order = list(range(len(mse.size())))
        per_ch_order[self.ch_axis] = 0
        per_ch_order[0] = self.ch_axis
        output = torch.flatten(mse.permute(per_ch_order), start_dim=1).mean(1)
        return output

    def reshape_as_ch_axis(self, x):
        per_ch_order = list(range(len(x.size())))
        per_ch_order[self.ch_axis] = 0
        per_ch_order[0] = self.ch_axis
        return torch.flatten(x.permute(per_ch_order), start_dim=1)

    def cal_loss(self, x, new_min, new_max):
        scale, offset = self._calculate_qparams(new_min, new_max)
        x_q = torch.fake_quantize_per_channel_affine(
            x,
            scale.data,
            offset.data.int(),
            self.ch_axis,
            self.quant_min,
            self.quant_max,
        )
        return self.mse_per_ch(x_q, x)

    def optim_range(self, x):
        x_min, x_max = torch.aminmax(self.reshape_as_ch_axis(x), dim=1)
        x_min_lower_limit = x_min.abs() * self.clipping_lower_limit
        x_max_lower_limit = x_max * self.clipping_lower_limit
        x_min_delta = x_min.abs() / self.steps * (1 - self.clipping_lower_limit)
        x_max_delta = x_max / self.steps * (1 - self.clipping_lower_limit)
        optimal_loss = torch.zeros_like(x_min) + 1e9

        for i in range(1, self.steps + 1):
            thres_min = x_min_lower_limit + (x_min_delta * i)
            thres_max = x_max_lower_limit + (x_max_delta * i)
            current_loss = self.cal_loss(x, -thres_min, thres_max)
            x_min = torch.where(current_loss < optimal_loss, -thres_min, x_min)
            x_max = torch.where(current_loss < optimal_loss, thres_max, x_max)
            optimal_loss = torch.min(current_loss, optimal_loss)
        return x_min, x_max

    def forward(self, x_orig):
        if not self.calibrated:
            x = x_orig.detach().to(self.min_val.dtype)
            self.min_val, self.max_val = self.optim_range(x)
            self.calibrated = True

        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        return torch.fake_quantize_per_channel_affine(
            x_orig,
            scale.data,
            zero_point.data.int(),
            self.ch_axis,
            self.quant_min,
            self.quant_max,
        )
