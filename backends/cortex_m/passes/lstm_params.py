# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ahead-of-time quantization math for the fused CMSIS-NN int8 LSTM kernel
(``arm_lstm_unidirectional_s8`` / ``arm_nn_lstm_step_s8``).

The CMSIS kernel does not consume observed gate-activation scales: the gate
pre-activation is fixed at Q3.12 (``2**-12``) and the sigmoid/tanh outputs at
Q0.15 (``2**-15``). Everything the kernel needs is derived here from four
quantities — the input activation scale, the output/hidden activation scale, a
chosen (power-of-two) cell-state scale, and per-gate weight scales — mirroring
CMSIS-NN's own generator (``Tests/UnitTest/RefactoredTestGen/Lib/op_lstm.py``).

``derive_lstm_params`` produces the CMSIS parameter structs (as a dataclass);
``quantized_lstm_reference`` is a faithful Python execution of the kernel used
both as the op's reference implementation and to validate the derivation
against a float ``torch.nn.LSTM``. PyTorch stores gates in IFGO order; CMSIS
addresses them by name (input/forget/cell/output).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from executorch.backends.cortex_m.passes.passes_utils import (
    quantize_multiplier_aot,
    requantize_cmsis,
)

# CMSIS-NN fixed intermediate Q-formats (see arm_nn_lstm_calculate_gate_s8_s16 /
# arm_nn_lstm_step_s8): gate accumulator is Q3.12, activation outputs are Q0.15.
_GATE_ACC_SCALE = 2.0**-12
_ACTIVATION_SCALE = 2.0**-15

_Q15_MIN = -32768
_Q15_MAX = 32767
_INT8_MIN = -128
_INT8_MAX = 127

# PyTorch weight_ih/weight_hh row order within the 4*H stack.
_IFGO = ("input", "forget", "cell", "output")


@dataclass
class GateParams:
    input_multiplier: int
    input_shift: int
    hidden_multiplier: int
    hidden_shift: int
    input_weights: torch.Tensor  # int8 [hidden, input]
    hidden_weights: torch.Tensor  # int8 [hidden, hidden]
    input_effective_bias: torch.Tensor  # int32 [hidden]
    hidden_effective_bias: torch.Tensor  # int32 [hidden]
    is_tanh: bool  # cell gate uses tanh; input/forget/output use sigmoid


@dataclass
class LstmParams:
    input_size: int
    hidden_size: int
    input_offset: int
    output_offset: int
    forget_to_cell_multiplier: int
    forget_to_cell_shift: int
    input_to_cell_multiplier: int
    input_to_cell_shift: int
    output_multiplier: int
    output_shift: int
    cell_scale_power: int
    cell_clip: int
    gates: dict[str, GateParams]  # keyed by "input"/"forget"/"cell"/"output"


def choose_cell_scale_power(max_abs_cell: float, headroom_bits: int = 2) -> int:
    """Pick the power-of-two cell-state scale exponent so the observed cell
    magnitude fits the int16 range with headroom for calibration drift.

    CMSIS stores the cell state as int16 with scale ``2**cell_scale_power`` and
    ``cell_clip = int16_max``; the scale must be a power of two (only the
    exponent is stored). ``ceil`` alone can leave as little as ~1x headroom, so
    inference values slightly above ``max_abs_cell`` would saturate; the
    ``headroom_bits`` margin (default 2 -> 4-8x) guards against that. Values that
    still exceed the range saturate at ``cell_clip`` rather than wrapping.
    """
    if max_abs_cell <= 0.0:
        return -15
    return math.ceil(math.log2(max_abs_cell / _Q15_MAX)) + headroom_bits


def _quantize_weight_per_tensor(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric per-tensor int8 quantization of a weight block."""
    max_abs = float(w.abs().max())
    scale = max_abs / _INT8_MAX if max_abs > 0 else 1.0
    q = torch.clamp(torch.round(w / scale), _INT8_MIN, _INT8_MAX).to(torch.int8)
    return q, scale


def _effective_bias(
    w_q: torch.Tensor, bias_q: torch.Tensor | None, offset: int
) -> torch.Tensor:
    """CMSIS effective bias = bias + offset * row_sum(weights) (int32)."""
    kernel_sum = w_q.to(torch.int32).sum(dim=1)
    eff = offset * kernel_sum
    if bias_q is not None:
        eff = eff + bias_q.to(torch.int32)
    return eff.to(torch.int32)


def derive_lstm_params(
    weight_ih: torch.Tensor,  # float [4H, input], IFGO row order
    weight_hh: torch.Tensor,  # float [4H, hidden]
    bias: torch.Tensor | None,  # float [4H] (b_ih + b_hh), IFGO
    input_scale: float,
    input_zp: int,
    output_scale: float,
    output_zp: int,
    cell_scale_power: int,
) -> LstmParams:
    """Derive the CMSIS-NN LSTM parameter set from float weights and the three
    boundary scales (input act, output/hidden act, chosen power-of-two cell).

    Single-layer, unidirectional only (``weight_ih`` is ``[4*hidden, input]``);
    the quantizer's pattern check rejects other configurations upstream.
    """
    four_h, input_size = weight_ih.shape
    if four_h % 4 != 0:
        raise ValueError(f"weight_ih first dim {four_h} is not 4*hidden_size")
    hidden_size = four_h // 4
    cell_scale = 2.0**cell_scale_power

    input_offset = -input_zp
    hidden_offset = -output_zp  # hidden state reuses the output scale/zp

    gates: dict[str, GateParams] = {}
    for gate_idx, name in enumerate(_IFGO):
        rows = slice(gate_idx * hidden_size, (gate_idx + 1) * hidden_size)
        w_ih_q, w_ih_scale = _quantize_weight_per_tensor(weight_ih[rows])
        w_hh_q, w_hh_scale = _quantize_weight_per_tensor(weight_hh[rows])

        bias_q = None
        if bias is not None:
            # Bias lives in the input-path accumulator domain.
            bias_acc_scale = input_scale * w_ih_scale
            bias_q = torch.round(bias[rows] / bias_acc_scale).to(torch.int32)

        in_mult, in_shift = quantize_multiplier_aot(
            input_scale * w_ih_scale / _GATE_ACC_SCALE
        )
        hid_mult, hid_shift = quantize_multiplier_aot(
            output_scale * w_hh_scale / _GATE_ACC_SCALE
        )

        gates[name] = GateParams(
            input_multiplier=in_mult,
            input_shift=in_shift,
            hidden_multiplier=hid_mult,
            hidden_shift=hid_shift,
            input_weights=w_ih_q,
            hidden_weights=w_hh_q,
            input_effective_bias=_effective_bias(w_ih_q, bias_q, input_offset),
            hidden_effective_bias=_effective_bias(w_hh_q, None, hidden_offset),
            is_tanh=(name == "cell"),
        )

    f2c_mult, f2c_shift = quantize_multiplier_aot(_ACTIVATION_SCALE)
    i2c_mult, i2c_shift = quantize_multiplier_aot(
        _ACTIVATION_SCALE * _ACTIVATION_SCALE / cell_scale
    )
    out_mult, out_shift = quantize_multiplier_aot(
        _ACTIVATION_SCALE * _ACTIVATION_SCALE / output_scale
    )

    return LstmParams(
        input_size=input_size,
        hidden_size=hidden_size,
        input_offset=input_offset,
        output_offset=output_zp,
        forget_to_cell_multiplier=f2c_mult,
        forget_to_cell_shift=f2c_shift,
        input_to_cell_multiplier=i2c_mult,
        input_to_cell_shift=i2c_shift,
        output_multiplier=out_mult,
        output_shift=out_shift,
        cell_scale_power=cell_scale_power,
        cell_clip=_Q15_MAX,
        gates=gates,
    )


def flatten_lstm_params(
    params: LstmParams,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    list[int],
    list[int],
]:
    """Stack the four per-gate tensors (IFGO order) into the flat form carried by
    the op schema: (input_weights[4H,in], hidden_weights[4H,H],
    input_effective_bias[4H], hidden_effective_bias[4H], and the four 4-element
    per-gate multiplier/shift lists for the input and hidden paths)."""
    g = [params.gates[n] for n in _IFGO]
    return (
        torch.cat([x.input_weights for x in g], dim=0),
        torch.cat([x.hidden_weights for x in g], dim=0),
        torch.cat([x.input_effective_bias for x in g], dim=0),
        torch.cat([x.hidden_effective_bias for x in g], dim=0),
        [x.input_multiplier for x in g],
        [x.input_shift for x in g],
        [x.hidden_multiplier for x in g],
        [x.hidden_shift for x in g],
    )


def lstm_params_from_op_args(
    input_weights: torch.Tensor,
    hidden_weights: torch.Tensor,
    input_effective_bias: torch.Tensor,
    hidden_effective_bias: torch.Tensor,
    input_multipliers: list[int],
    input_shifts: list[int],
    hidden_multipliers: list[int],
    hidden_shifts: list[int],
    input_offset: int,
    output_offset: int,
    forget_to_cell_multiplier: int,
    forget_to_cell_shift: int,
    input_to_cell_multiplier: int,
    input_to_cell_shift: int,
    output_multiplier: int,
    output_shift: int,
    cell_scale_power: int,
    cell_clip: int,
) -> LstmParams:
    """Rebuild the ``LstmParams`` struct from the op's flat args (inverse of
    ``flatten_lstm_params`` plus the scalar fields carried directly)."""
    four_h, input_size = input_weights.shape
    hidden_size = four_h // 4
    gates = {}
    for gate_idx, name in enumerate(_IFGO):
        rows = slice(gate_idx * hidden_size, (gate_idx + 1) * hidden_size)
        gates[name] = GateParams(
            input_multiplier=int(input_multipliers[gate_idx]),
            input_shift=int(input_shifts[gate_idx]),
            hidden_multiplier=int(hidden_multipliers[gate_idx]),
            hidden_shift=int(hidden_shifts[gate_idx]),
            input_weights=input_weights[rows],
            hidden_weights=hidden_weights[rows],
            input_effective_bias=input_effective_bias[rows],
            hidden_effective_bias=hidden_effective_bias[rows],
            is_tanh=(name == "cell"),
        )
    return LstmParams(
        input_size=input_size,
        hidden_size=hidden_size,
        input_offset=input_offset,
        output_offset=output_offset,
        forget_to_cell_multiplier=forget_to_cell_multiplier,
        forget_to_cell_shift=forget_to_cell_shift,
        input_to_cell_multiplier=input_to_cell_multiplier,
        input_to_cell_shift=input_to_cell_shift,
        output_multiplier=output_multiplier,
        output_shift=output_shift,
        cell_scale_power=cell_scale_power,
        cell_clip=cell_clip,
        gates=gates,
    )


def _activation_q15(
    x_q16: torch.Tensor, real_scale: float, is_tanh: bool
) -> torch.Tensor:
    """Faithful stand-in for arm_nn_activation_s16: interpret the int16 input at
    ``real_scale``, apply sigmoid/tanh in float, emit Q0.15 int16. This is the
    exact operation the CMSIS table approximates (to ~1 LSB)."""
    x = x_q16.to(torch.float64) * real_scale
    y = torch.tanh(x) if is_tanh else torch.sigmoid(x)
    q = torch.round(y / _ACTIVATION_SCALE)
    return torch.clamp(q, _Q15_MIN, _Q15_MAX).to(torch.int32)


def _gate(
    x_q: torch.Tensor,  # int8 [input]
    h_q: torch.Tensor | None,  # int8 [hidden] or None at t=0
    gp: GateParams,
) -> torch.Tensor:
    """One gate: two accumulating int8xint8->int16 matmuls + activation (Q0.15)."""
    acc = gp.input_effective_bias + (
        x_q.to(torch.int32) * gp.input_weights.to(torch.int32)
    ).sum(dim=1)
    out = torch.clamp(
        requantize_cmsis(acc, gp.input_multiplier, gp.input_shift), _Q15_MIN, _Q15_MAX
    )
    if h_q is not None:
        acc_h = gp.hidden_effective_bias + (
            h_q.to(torch.int32) * gp.hidden_weights.to(torch.int32)
        ).sum(dim=1)
        out = torch.clamp(
            requantize_cmsis(acc_h, gp.hidden_multiplier, gp.hidden_shift) + out,
            _Q15_MIN,
            _Q15_MAX,
        )
    return _activation_q15(out, _GATE_ACC_SCALE, gp.is_tanh)


def run_quantized_lstm(params: LstmParams, x_q: torch.Tensor) -> torch.Tensor:
    """Core int8 CMSIS LSTM: int8 input [T, B, input] -> int8 output
    [T, B, hidden], full sequence, zero initial state, time-major. Shared by the
    op's reference implementation and the float-model validation wrapper.
    """
    T, B, _ = x_q.shape
    H = params.hidden_size
    cell_scale = 2.0**params.cell_scale_power
    x_q = x_q.to(torch.int32)

    out = torch.empty(T, B, H, dtype=torch.int32)
    for b in range(B):
        h_q: torch.Tensor | None = None
        cell = torch.zeros(H, dtype=torch.int32)
        for t in range(T):
            xt = x_q[t, b]
            forget = _gate(xt, h_q, params.gates["forget"])
            # cell = clamp(requant(forget * cell), Q15) -- overwrite
            cell = torch.clamp(
                requantize_cmsis(
                    forget * cell,
                    params.forget_to_cell_multiplier,
                    params.forget_to_cell_shift,
                ),
                _Q15_MIN,
                _Q15_MAX,
            )
            inp = _gate(xt, h_q, params.gates["input"])
            cellg = _gate(xt, h_q, params.gates["cell"])
            # cell += clamp(requant(inp * cellg), [-cell_clip, cell_clip])
            cell = torch.clamp(
                requantize_cmsis(
                    inp * cellg,
                    params.input_to_cell_multiplier,
                    params.input_to_cell_shift,
                )
                + cell,
                -params.cell_clip,
                params.cell_clip,
            )
            outp = _gate(xt, h_q, params.gates["output"])
            tanh_cell = _activation_q15(cell, cell_scale, is_tanh=True)
            hidden = (
                requantize_cmsis(
                    outp * tanh_cell,
                    params.output_multiplier,
                    params.output_shift,
                )
                + params.output_offset
            )
            hidden_q = torch.clamp(hidden, _INT8_MIN, _INT8_MAX).to(torch.int32)
            out[t, b] = hidden_q
            h_q = hidden_q
    return out.to(torch.int8)


def quantized_lstm_reference(
    x: torch.Tensor,  # float input, [T, B, input] (time-major)
    params: LstmParams,
    input_scale: float,
    output_scale: float,
    dequantize: bool = False,
) -> torch.Tensor:
    """Validation wrapper: quantize a float input, run the int8 LSTM, and
    optionally dequantize the output for comparison against a float model.
    """
    input_zp = -params.input_offset
    x_q = torch.clamp(torch.round(x / input_scale) + input_zp, _INT8_MIN, _INT8_MAX)
    out8 = run_quantized_lstm(params, x_q)
    if dequantize:
        return (out8.to(torch.int32) - params.output_offset).to(
            torch.float32
        ) * output_scale
    return out8
