# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the CMSIS-NN LSTM AoT math: the derived parameters + the
int8 reference implementation must track a float torch.nn.LSTM within the
quantization noise floor."""

from typing import cast

import torch
import torch.nn as nn
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.passes.lstm_params import (
    choose_cell_scale_power,
    derive_lstm_params,
    quantized_lstm_reference,
)


def _affine_int8_qparams(t: torch.Tensor) -> tuple[float, int]:
    lo = min(0.0, float(t.min()))
    hi = max(0.0, float(t.max()))
    scale = (hi - lo) / 255.0 if hi > lo else 1e-6
    zp = max(-128, min(127, int(round(-128 - lo / scale))))
    return scale, zp


def _lstm_weights(
    lstm: nn.LSTM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # nn.Module.__getattr__ types these as Tensor | Module; they are tensors.
    weight_ih = cast(torch.Tensor, lstm.weight_ih_l0)
    weight_hh = cast(torch.Tensor, lstm.weight_hh_l0)
    bias = cast(torch.Tensor, lstm.bias_ih_l0) + cast(torch.Tensor, lstm.bias_hh_l0)
    return weight_ih, weight_hh, bias


def _float_cell_max_abs(x: torch.Tensor, lstm: nn.LSTM) -> float:
    h_size = lstm.hidden_size
    weight_ih, weight_hh, layer_bias = _lstm_weights(lstm)
    w_ih, w_hh, bias = weight_ih.detach(), weight_hh.detach(), layer_bias.detach()
    time_steps, batch = x.shape[0], x.shape[1]
    max_abs = 0.0
    for b in range(batch):
        h = torch.zeros(h_size)
        c = torch.zeros(h_size)
        for t in range(time_steps):
            g = x[t, b] @ w_ih.T + h @ w_hh.T + bias
            i, f = torch.sigmoid(g[0:h_size]), torch.sigmoid(g[h_size : 2 * h_size])
            gg, o = torch.tanh(g[2 * h_size : 3 * h_size]), torch.sigmoid(
                g[3 * h_size :]
            )
            c = f * c + i * gg
            h = o * torch.tanh(c)
            max_abs = max(max_abs, float(c.abs().max()))
    return max_abs


def _sqnr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    noise = (ref - test).pow(2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10(ref.pow(2).mean() / noise))


shapes = {
    "small": (4, 3, 8),
    "wide": (8, 16, 12),
    "single_step": (4, 5, 1),
}


@parametrize("shape", shapes)
def test_reference_tracks_float_lstm(shape: tuple[int, int, int]) -> None:
    input_size, hidden_size, time_steps = shape
    lstm = nn.LSTM(input_size, hidden_size).eval()
    x = torch.randn(time_steps, 1, input_size)
    with torch.no_grad():
        y_float, _ = lstm(x)

    input_scale, input_zp = _affine_int8_qparams(x)
    output_scale, output_zp = _affine_int8_qparams(y_float)
    cell_scale_power = choose_cell_scale_power(_float_cell_max_abs(x, lstm))

    weight_ih, weight_hh, bias = _lstm_weights(lstm)
    params = derive_lstm_params(
        weight_ih.detach(),
        weight_hh.detach(),
        bias.detach(),
        input_scale,
        input_zp,
        output_scale,
        output_zp,
        cell_scale_power,
    )
    y_ref = quantized_lstm_reference(
        x, params, input_scale, output_scale, dequantize=True
    )

    assert _sqnr_db(y_float, y_ref) > 30.0
