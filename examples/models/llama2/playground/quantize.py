
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.quantized_ops import *  # noqa
from ..quantize import _check_linear_int4_k, find_multiple # noqa


def get_group_qparams_symmetric(w, n_bit=4, groupsize=32, precision=torch.float32):
    # GGML Q4_0 quantization.
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    max_val_abs = torch.max(-min_val_neg, max_val_pos)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))

    scales = max_val_abs / (float(min_int - max_int) / 2) # for 4 bit this is max / -8
    scales = torch.min(scales, torch.full_like(scales, -torch.finfo(precision).eps)) # scale can't be larger than -eps
    # TODO: make sure abs(scales) is not too small?
    zeros = torch.full_like(scales, 8.5)
    return scales.to(precision).reshape(w.shape[0], -1), zeros.to(precision).reshape(
        w.shape[0], -1
    )


def group_quantize_tensor_symmetric(
    w, n_bit=4, group_size=32, precision=torch.float32
):
    scales, zeros = get_group_qparams_symmetric(w, n_bit, group_size, precision)
    n_bit = 4
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    # TODO: currently we don't know how to express torch.int4, we'll
    # add torch.int4 to core later
    w_int8 = torch.ops.quantized_decomposed.quantize_per_channel_group(
        w, scales, zeros, min_int, max_int, torch.int8, group_size
    )

    return w_int8, scales, zeros

def prepare_int4_weight_and_scales_and_zeros(weight, group_size, precision):
    """
    llama.cpp Q4_0 quantization scheme. Symmetric groupwise 4bit quant with group
    size 32 and zero point being fixed to 8.5.
    """
    weight_int8, scales, zeros = group_quantize_tensor_symmetric(
        weight,
        n_bit=4,
        group_size=group_size,
        precision=precision,
    )
    # weight_int4packed = torch.ops.quantized_decomposed.pack_int4_from_int8(weight_int8)
    return weight_int8, scales, zeros



def replace_linear_4w(
    module,
    group_size,
    padding_allowed,
    precision,
    scales_precision,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _check_linear_int4_k(child.in_features, group_size) or padding_allowed:
                setattr(
                    module,
                    name,
                    Int8DynActInt4WeightLinear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        group_size=group_size,
                        precision=precision,
                        scales_precision=scales_precision,
                    ),
                )
        else:
            replace_linear_4w(
                child,
                group_size,
                padding_allowed,
                precision,
                scales_precision,
            )


class Int8DynActInt4WeightQuantHandler:
    def __init__(
        self,
        mod,
        group_size=32,
        padding_allowed=False,
        precision=torch.float32,
        scales_precision=torch.float32,
    ):
        self.mod = mod
        self.group_size = group_size
        self.padding_allowed = padding_allowed
        self.precision = precision
        self.scales_precision = scales_precision
        # assert group_size in [32, 64, 128, 256]

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                print("in features:", in_features, " out features:", out_features)
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                print(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert (
                    in_features % self.group_size == 0
                ), f"require in_features:{in_features} % self.group_size:{self.group_size} == 0"

                weight = mod.weight.data
                """
                if not _check_linear_int4_k(
                    in_features, self.group_size
                ):
                    if self.padding_allowed:
                        print(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = _calc_padded_size_linear_int4(
                            in_features, self.group_size
                        )
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        raise RuntimeError(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that group_size"
                        )
                """
                (
                    weight_int4pack,
                    scales,
                    zeros,
                ) = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(self.precision),
                    self.group_size,
                    self.scales_precision,
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales"] = scales.to("cpu")
                cur_state_dict[f"{fqn}.zeros"] = zeros.to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_4w(
            self.mod,
            self.group_size,
            self.padding_allowed,
            self.precision,
            self.scales_precision,
        )
        return self.mod

    def quantized_model(self) -> nn.Module:
        model_updated_state_dict = self.create_quantized_state_dict()
        self.convert_for_runtime()
        self.mod.load_state_dict(model_updated_state_dict)
        return self.mod


class Int4WeightLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int4 weight.
    Weights are per channel groupwise quantized. Activations will be quantized
    into int8 in custom op.

    Parameters of importance:

    group_size: the number of elements in each quantized group
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    scales_precision: precision of per group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        group_size: int = 32,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # always pad if needed since it becomes a noop at runtime if not needed
        # self.origin_in_features = in_features
        assert (
            in_features % group_size == 0
        ), f"require in_features:{in_features} % group_size:{group_size} == 0"
        # in_features = _calc_padded_size_linear_int4(
        #    in_features, group_size
        # )
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.group_size = group_size
        # Precision of the activation which also indicates
        # output precision of the dynamically quantized linear layer
        # that his module represents.
        self.precision = precision

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=scales_precision,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=scales_precision,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        # Change this to pad if needed later
        # else this op will always show up
        # input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))

        """
        TODO: add a custom op here that takes quantized weights (int4 but unpacked into int8)
        and fp32 activation, return fp32 result. Inside the op will convert activation to
        int8, weights to int4 and perform dot product on int4 and int8.

        ggml::linear_q4_0(Tensor weights, Tensor scale, Tensor zeros, Tensor activation) -> Tensor

        """
        return torch.ops.ggml.linear_q4_0(
            self.weight, self.scales, self.zeros, input
        ).to(self.precision)
