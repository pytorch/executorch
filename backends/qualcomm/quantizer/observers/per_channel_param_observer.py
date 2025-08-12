# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchao.quantization.pt2e import UniformQuantizationObserverBase


# TODO move to torch/ao/quantization/observer.py.
class PerChannelParamObserver(UniformQuantizationObserverBase):
    """
    Minimize quantization loss caused by outlier via linear search. More details can be found at https://arxiv.org/pdf/2209.13325
    """

    def __init__(
        self,
        ch_axis=0,
        use_mse=True,
        steps=100,
        dtype=torch.int8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,  # noqa: B008
        is_dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(
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

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.ch_axis = ch_axis
        self.use_mse = use_mse
        self.steps = steps
        self.calibrated = False

    def to_ch_axis(self, x):
        axis_order = list(range(len(x.size())))
        axis_order[self.ch_axis], axis_order[0] = 0, self.ch_axis
        return torch.flatten(x.permute(axis_order), start_dim=1)

    def mse(self, pred, expect):
        loss = (pred - expect).abs().pow(2)
        return self.to_ch_axis(loss).mean(1)

    def cosine(self, pred, expect):
        target = torch.ones(pred.shape[self.ch_axis])
        pred_n = self.to_ch_axis(pred).reshape(pred.shape[0], -1)
        expect_n = self.to_ch_axis(expect).reshape(expect.shape[0], -1)
        return torch.nn.CosineEmbeddingLoss()(pred_n, expect_n, target)

    def loss_fn(self, x, new_min, new_max):
        scale, offset = self._calculate_qparams(new_min, new_max)
        x_q = torch.fake_quantize_per_channel_affine(
            x,
            scale.data,
            offset.data.int(),
            self.ch_axis,
            self.quant_min,
            self.quant_max,
        )
        return self.mse(x_q, x) if self.use_mse else self.cosine(x_q, x)

    def line_search(self, x):
        x_min, x_max = torch.aminmax(self.to_ch_axis(x), dim=1)
        x_range = torch.max(x_min.abs(), x_max)
        optimal_loss = torch.zeros_like(x_min) + 1e9

        # check which clip range could produce smallest loss
        for i in range(1, self.steps + 1):
            thres = x_range / self.steps * i
            current_loss = self.loss_fn(x, -thres, thres)
            x_min = torch.where(current_loss < optimal_loss, -thres, x_min)
            x_max = torch.where(current_loss < optimal_loss, thres, x_max)
            optimal_loss = torch.min(current_loss, optimal_loss)

        return x_min, x_max

    def forward(self, x_orig):
        # since params are static, one calibration is enough
        if not self.calibrated:
            x = x_orig.detach().to(self.min_val.dtype)
            self.min_val, self.max_val = self.line_search(x)
            self.calibrated = True

        # return fake-quant result for saturating outliers
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        return torch.fake_quantize_per_channel_affine(
            x_orig,
            scale.data,
            zero_point.data.int(),
            self.ch_axis,
            self.quant_min,
            self.quant_max,
        )

    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)
