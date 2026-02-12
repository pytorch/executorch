# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantizable GRU modules following the torch.ao.nn.quantizable.LSTM pattern.

The standard nn.GRU is an opaque composite op that the quantizer cannot
annotate. This module decomposes GRU into nn.Linear + FloatFunctional
so that QAT observers can be inserted at each arithmetic boundary.

GRU cell equations:
    r_t = sigmoid(x_t @ W_ir.T + b_ir + h_{t-1} @ W_hr.T + b_hr)
    z_t = sigmoid(x_t @ W_iz.T + b_iz + h_{t-1} @ W_hz.T + b_hz)
    n_t = tanh(x_t @ W_in.T + b_in + r_t * (h_{t-1} @ W_hn.T + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    """A quantizable GRU cell with FloatFunctional ops for each arithmetic boundary."""

    _FLOAT_MODULE = nn.GRUCell

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Input projections: x_t -> [r, z, n] (3*hidden_size)
        self.input_linear = nn.Linear(
            input_size, 3 * hidden_size, bias=bias, **factory_kwargs
        )
        # Hidden projections: h_{t-1} -> [r, z, n] (3*hidden_size)
        self.hidden_linear = nn.Linear(
            hidden_size, 3 * hidden_size, bias=bias, **factory_kwargs
        )

        # Gate activations
        self.reset_gate = nn.Sigmoid()
        self.update_gate = nn.Sigmoid()
        self.new_gate = nn.Tanh()

        # FloatFunctional for each observable arithmetic op
        self.add_r = torch.ao.nn.quantized.FloatFunctional()  # input_r + hidden_r
        self.add_z = torch.ao.nn.quantized.FloatFunctional()  # input_z + hidden_z
        self.mul_r_nh = torch.ao.nn.quantized.FloatFunctional()  # r_t * hidden_n
        self.add_n = torch.ao.nn.quantized.FloatFunctional()  # input_n + r*hidden_n
        self.mul_1mz_n = torch.ao.nn.quantized.FloatFunctional()  # (1-z) * n
        self.mul_z_h = torch.ao.nn.quantized.FloatFunctional()  # z * h_{t-1}
        self.add_h = torch.ao.nn.quantized.FloatFunctional()  # (1-z)*n + z*h

    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_size, device=x.device)

        igates = self.input_linear(x)
        hgates = self.hidden_linear(hidden)

        # Split into r, z, n components
        H = self.hidden_size
        input_r, input_z, input_n = (
            igates[:, :H],
            igates[:, H : 2 * H],
            igates[:, 2 * H :],
        )
        hidden_r, hidden_z, hidden_n = (
            hgates[:, :H],
            hgates[:, H : 2 * H],
            hgates[:, 2 * H :],
        )

        r_t = self.reset_gate(self.add_r.add(input_r, hidden_r))
        z_t = self.update_gate(self.add_z.add(input_z, hidden_z))
        n_t = self.new_gate(self.add_n.add(input_n, self.mul_r_nh.mul(r_t, hidden_n)))

        h_t = self.add_h.add(
            self.mul_1mz_n.mul(1.0 - z_t, n_t),
            self.mul_z_h.mul(z_t, hidden),
        )
        return h_t

    @classmethod
    def from_params(
        cls,
        wi: Tensor,
        wh: Tensor,
        bi: Optional[Tensor] = None,
        bh: Optional[Tensor] = None,
    ) -> "GRUCell":
        input_size = wi.shape[1]
        hidden_size = wh.shape[1]
        cell = cls(input_size, hidden_size, bias=(bi is not None))
        cell.input_linear.weight = nn.Parameter(wi)
        if bi is not None:
            cell.input_linear.bias = nn.Parameter(bi)
        cell.hidden_linear.weight = nn.Parameter(wh)
        if bh is not None:
            cell.hidden_linear.bias = nn.Parameter(bh)
        return cell

    @classmethod
    def from_float(cls, other, use_precomputed_fake_quant=False):
        assert type(other) is cls._FLOAT_MODULE
        assert hasattr(other, "qconfig"), "The float module must have 'qconfig'"
        observed = cls.from_params(
            other.weight_ih,
            other.weight_hh,
            other.bias_ih,
            other.bias_hh,
        )
        observed.qconfig = other.qconfig
        observed.input_linear.qconfig = other.qconfig
        observed.hidden_linear.qconfig = other.qconfig
        return observed


class _GRUSingleLayer(nn.Module):
    """A single one-directional GRU layer that processes a sequence."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size, bias=bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        result = []
        seq_len = x.shape[0]
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for i in indices:
            hidden = self.cell(x[i], hidden)
            result.append(hidden)
        if reverse:
            result.reverse()
        return torch.stack(result, 0), hidden

    @classmethod
    def from_params(cls, *args, **kwargs):
        cell = GRUCell.from_params(*args, **kwargs)
        layer = cls(cell.input_size, cell.hidden_size, cell.bias)
        layer.cell = cell
        return layer


class _GRULayer(nn.Module):
    """A single bi-directional GRU layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.layer_fw = _GRUSingleLayer(
            input_size, hidden_size, bias=bias, **factory_kwargs
        )
        if self.bidirectional:
            self.layer_bw = _GRUSingleLayer(
                input_size, hidden_size, bias=bias, **factory_kwargs
            )

    def forward(
        self, x: Tensor, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        hx_fw = None
        hx_bw = None
        if hidden is not None:
            if self.bidirectional:
                hx_fw = hidden[0]
                hx_bw = hidden[1]
            else:
                hx_fw = hidden

        result_fw, h_fw = self.layer_fw(x, hx_fw)

        if self.bidirectional:
            result_bw, h_bw = self.layer_bw(x, hx_bw, reverse=True)
            result = torch.cat([result_fw, result_bw], result_fw.dim() - 1)
            h = torch.stack([h_fw, h_bw], 0)
        else:
            result = result_fw
            h = h_fw

        if self.batch_first:
            result = result.transpose(0, 1)

        return result, h

    @classmethod
    def from_float(cls, other, layer_idx=0, qconfig=None, **kwargs):
        assert hasattr(other, "qconfig") or (qconfig is not None)

        input_size = kwargs.get("input_size", other.input_size)
        hidden_size = kwargs.get("hidden_size", other.hidden_size)
        bias = kwargs.get("bias", other.bias)
        batch_first = kwargs.get("batch_first", other.batch_first)
        bidirectional = kwargs.get("bidirectional", other.bidirectional)

        layer = cls(input_size, hidden_size, bias, batch_first, bidirectional)
        layer.qconfig = getattr(other, "qconfig", qconfig)

        wi = getattr(other, f"weight_ih_l{layer_idx}")
        wh = getattr(other, f"weight_hh_l{layer_idx}")
        bi = getattr(other, f"bias_ih_l{layer_idx}", None)
        bh = getattr(other, f"bias_hh_l{layer_idx}", None)
        layer.layer_fw = _GRUSingleLayer.from_params(wi, wh, bi, bh)

        if other.bidirectional:
            wi = getattr(other, f"weight_ih_l{layer_idx}_reverse")
            wh = getattr(other, f"weight_hh_l{layer_idx}_reverse")
            bi = getattr(other, f"bias_ih_l{layer_idx}_reverse", None)
            bh = getattr(other, f"bias_hh_l{layer_idx}_reverse", None)
            layer.layer_bw = _GRUSingleLayer.from_params(wi, wh, bi, bh)
        return layer


class GRU(nn.Module):
    """A quantizable GRU following the torch.ao.nn.quantizable.LSTM pattern.

    Converts a standard nn.GRU into observable form with nn.Linear +
    FloatFunctional ops for each arithmetic boundary.
    """

    _FLOAT_MODULE = nn.GRU

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.training = False

        num_directions = 2 if bidirectional else 1
        layers: List[_GRULayer] = [
            _GRULayer(
                input_size,
                hidden_size,
                bias,
                batch_first=False,
                bidirectional=bidirectional,
                **factory_kwargs,
            )
        ]
        for _ in range(1, num_layers):
            layers.append(
                _GRULayer(
                    hidden_size * num_directions,
                    hidden_size,
                    bias,
                    batch_first=False,
                    bidirectional=bidirectional,
                    **factory_kwargs,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: Tensor, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx_list = [None] * self.num_layers
        else:
            hx = hidden.reshape(
                self.num_layers, num_directions, hidden.shape[-2], hidden.shape[-1]
            )
            hx_list = [hx[idx].squeeze(0) for idx in range(self.num_layers)]

        h_list = []
        for idx, layer in enumerate(self.layers):
            x, h = layer(x, hx_list[idx])
            h_list.append(h)

        h_tensor = torch.stack(h_list)
        h_tensor = h_tensor.reshape(-1, h_tensor.shape[-2], h_tensor.shape[-1])

        if self.batch_first:
            x = x.transpose(0, 1)

        return x, h_tensor

    @classmethod
    def from_float(cls, other, qconfig=None):
        assert isinstance(other, cls._FLOAT_MODULE)
        assert hasattr(other, "qconfig") or qconfig
        observed = cls(
            other.input_size,
            other.hidden_size,
            other.num_layers,
            other.bias,
            other.batch_first,
            other.dropout,
            other.bidirectional,
        )
        observed.qconfig = getattr(other, "qconfig", qconfig)
        for idx in range(other.num_layers):
            observed.layers[idx] = _GRULayer.from_float(
                other, idx, qconfig, batch_first=False
            )
        if other.training:
            observed.train()
            observed = torch.ao.quantization.prepare_qat(observed, inplace=True)
        else:
            observed.eval()
            observed = torch.ao.quantization.prepare(observed, inplace=True)
        return observed
