# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchao.quantization.pt2e import PerChannelMinMaxObserver


_EPS = torch.finfo(torch.float32).eps


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
