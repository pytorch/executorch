# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantizable RNN modules following the torch.ao.nn.quantizable.LSTM pattern.

The standard nn.RNN is an opaque composite op that the quantizer cannot
annotate. This module decomposes RNN into nn.Linear + FloatFunctional
so that QAT observers can be inserted at each arithmetic boundary.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor


class RNNCell(nn.Module):
    """A quantizable RNN cell.

    Equation: h_t = activation(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)

    Uses nn.Linear for projections and FloatFunctional for the addition,
    making each arithmetic op observable for quantization.
    """

    _FLOAT_MODULE = nn.RNNCell

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.input_linear = nn.Linear(
            input_size, hidden_size, bias=bias, **factory_kwargs
        )
        self.hidden_linear = nn.Linear(
            hidden_size, hidden_size, bias=bias, **factory_kwargs
        )
        self.add_gates = torch.ao.nn.quantized.FloatFunctional()

        if nonlinearity == "tanh":
            self.activation = nn.Tanh()
        elif nonlinearity == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        gates = self.add_gates.add(self.input_linear(x), self.hidden_linear(hidden))
        return self.activation(gates)

    @classmethod
    def from_params(
        cls,
        wi: Tensor,
        wh: Tensor,
        bi: Optional[Tensor] = None,
        bh: Optional[Tensor] = None,
        nonlinearity: str = "tanh",
    ) -> "RNNCell":
        input_size = wi.shape[1]
        hidden_size = wh.shape[1]
        cell = cls(
            input_size, hidden_size, bias=(bi is not None), nonlinearity=nonlinearity
        )
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
            nonlinearity=other.nonlinearity,
        )
        observed.qconfig = other.qconfig
        observed.input_linear.qconfig = other.qconfig
        observed.hidden_linear.qconfig = other.qconfig
        return observed


class _RNNSingleLayer(nn.Module):
    """A single one-directional RNN layer that processes a sequence."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.cell = RNNCell(
            input_size,
            hidden_size,
            bias=bias,
            nonlinearity=nonlinearity,
            **factory_kwargs,
        )

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
        cell = RNNCell.from_params(*args, **kwargs)
        layer = cls(cell.input_size, cell.hidden_size, cell.bias, cell.nonlinearity)
        layer.cell = cell
        return layer


class _RNNLayer(nn.Module):
    """A single bi-directional RNN layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.layer_fw = _RNNSingleLayer(
            input_size,
            hidden_size,
            bias=bias,
            nonlinearity=nonlinearity,
            **factory_kwargs,
        )
        if self.bidirectional:
            self.layer_bw = _RNNSingleLayer(
                input_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                **factory_kwargs,
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
        nonlinearity = kwargs.get("nonlinearity", other.nonlinearity)

        layer = cls(
            input_size,
            hidden_size,
            bias,
            batch_first,
            bidirectional,
            nonlinearity=nonlinearity,
        )
        layer.qconfig = getattr(other, "qconfig", qconfig)

        wi = getattr(other, f"weight_ih_l{layer_idx}")
        wh = getattr(other, f"weight_hh_l{layer_idx}")
        bi = getattr(other, f"bias_ih_l{layer_idx}", None)
        bh = getattr(other, f"bias_hh_l{layer_idx}", None)
        layer.layer_fw = _RNNSingleLayer.from_params(
            wi, wh, bi, bh, nonlinearity=nonlinearity
        )

        if other.bidirectional:
            wi = getattr(other, f"weight_ih_l{layer_idx}_reverse")
            wh = getattr(other, f"weight_hh_l{layer_idx}_reverse")
            bi = getattr(other, f"bias_ih_l{layer_idx}_reverse", None)
            bh = getattr(other, f"bias_hh_l{layer_idx}_reverse", None)
            layer.layer_bw = _RNNSingleLayer.from_params(
                wi, wh, bi, bh, nonlinearity=nonlinearity
            )
        return layer


class RNN(nn.Module):
    """A quantizable RNN following the torch.ao.nn.quantizable.LSTM pattern.

    Converts a standard nn.RNN into observable form with nn.Linear +
    FloatFunctional ops for each arithmetic boundary.
    """

    _FLOAT_MODULE = nn.RNN

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
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
        self.nonlinearity = nonlinearity
        self.training = False

        num_directions = 2 if bidirectional else 1
        layers: List[_RNNLayer] = [
            _RNNLayer(
                input_size,
                hidden_size,
                bias,
                batch_first=False,
                bidirectional=bidirectional,
                nonlinearity=nonlinearity,
                **factory_kwargs,
            )
        ]
        for _ in range(1, num_layers):
            layers.append(
                _RNNLayer(
                    hidden_size * num_directions,
                    hidden_size,
                    bias,
                    batch_first=False,
                    bidirectional=bidirectional,
                    nonlinearity=nonlinearity,
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
            nonlinearity=other.nonlinearity,
        )
        observed.qconfig = getattr(other, "qconfig", qconfig)
        for idx in range(other.num_layers):
            observed.layers[idx] = _RNNLayer.from_float(
                other, idx, qconfig, batch_first=False
            )
        if other.training:
            observed.train()
            observed = torch.ao.quantization.prepare_qat(observed, inplace=True)
        else:
            observed.eval()
            observed = torch.ao.quantization.prepare(observed, inplace=True)
        return observed
