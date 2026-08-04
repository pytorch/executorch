# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Helper functions for tranforming the model to be able to load pre-quantized checkpoints.

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchao.quantization.linear_quant_modules import (
    _check_linear_int4_k,
    Int8DynActInt4WeightLinear,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.quant_primitives import dequantize_affine

from .quantize import Int8DynActInt8WeightLinear, QuantizedGroupEmbedding


class WeightOnlyInt4Linear(torch.nn.Module):
    """Weight-only int4 per-group linear that reuses the 8da4w checkpoint layout.

    Stores the SAME buffers as ``Int8DynActInt4WeightLinear`` (int8 ``weight`` holding
    int4 values, per-group ``scales``/``zeros``) so a pre-quantized QAT/PTQ checkpoint
    loads unchanged. The difference is forward: the activation is left in floating
    point (no per-token dynamic quant), so the traced graph is
    ``dequantize_affine(weight) -> F.linear`` with no activation quant. This lowers to
    the ET-VK weight-only ``et_vk.linear_q4gsw`` path instead of the dynamic-activation
    ``et_vk.linear_dq8ca_q4gsw`` path.
    """

    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert (
            in_features % groupsize == 0
        ), f"require in_features:{in_features} % groupsize:{groupsize} == 0"
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.precision = precision
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
                device=device,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (out_features, in_features // groupsize),
                dtype=scales_precision,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=precision, device=device)
            )
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.precision)
        n_bit = 4
        quant_min = -(2 ** (n_bit - 1))
        quant_max = 2 ** (n_bit - 1) - 1
        block_size = (1, self.groupsize)
        w_dq = dequantize_affine(
            self.weight,
            block_size,
            self.scales,
            self.zeros,
            torch.int8,
            quant_min,
            quant_max,
            output_dtype=self.precision,
        )
        return F.linear(input, w_dq, self.bias)


def _replace_linear_with_linear_int4_weight_only_for_pre_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    group_size: int,
    precision: torch.dtype,
    scales_precision: torch.dtype,
):
    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        scales_key = f"{cur_fqn}.scales"
        if isinstance(child, nn.Linear) and scales_key in checkpoint:
            assert _check_linear_int4_k(child.in_features, group_size)
            assert checkpoint[f"{cur_fqn}.weight"].dtype == torch.int8
            assert checkpoint[scales_key].dtype == scales_precision
            return True
        return False

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = WeightOnlyInt4Linear(
            # pyre-fixme[6]
            child.in_features,
            # pyre-fixme[6]
            child.out_features,
            bias=child.bias is not None,
            device=child.weight.device,
            groupsize=group_size,
            precision=precision,
            scales_precision=scales_precision,
        )
        # Symmetric int4: zero point is 0 (matches the 8da4w pre-quant path, which
        # zeros this buffer rather than trusting the checkpoint's zeros key).
        new_linear.zeros = torch.zeros_like(new_linear.zeros)
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def _replace_linear_with_linear_8da4w_for_pre_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    group_size: int,
    precision: torch.dtype,
    scales_precision: torch.dtype,
):
    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        # Only replace linear layers where the checkpoint contains explicit scales
        scales_key = f"{cur_fqn}.scales"
        if isinstance(child, nn.Linear) and scales_key in checkpoint:
            assert _check_linear_int4_k(child.in_features, group_size)
            assert checkpoint[f"{cur_fqn}.weight"].dtype == torch.int8
            assert checkpoint[scales_key].dtype == scales_precision
            return True
        return False

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = Int8DynActInt4WeightLinear(
            # pyre-fixme[6]: For 1st argument expected `int` but got `Union[Module,
            #  Tensor]`.
            child.in_features,
            # pyre-fixme[6]: For 2nd argument expected `int` but got `Union[Module,
            #  Tensor]`.
            child.out_features,
            bias=child.bias is not None,
            device=child.weight.device,
            groupsize=group_size,
            precision=precision,
            scales_precision=scales_precision,
        )
        # TODO(lunwenh): Remove this once TorchAO's commit pin in ExecuTorch is updated to include this PR
        new_linear.zeros = torch.zeros_like(new_linear.zeros)
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def transform_linear_for_pre_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    group_size: int,
    dtype: torch.dtype,
    weight_only: bool = False,
) -> torch.nn.Module:
    """
    Transform the model to be able to load pre-quantized checkpoints that
    are quantized with the given group size and quantization mode for
    linear layers.

    When ``weight_only`` is True, linears are swapped for ``WeightOnlyInt4Linear``
    (float activation x dequantized int4 weight) instead of the default
    ``Int8DynActInt4WeightLinear`` (dynamic per-token int8 activation). The
    checkpoint buffers (int4 weight, per-group scales/zeros) are identical, so the
    same pre-quantized checkpoint loads either way.
    """

    if group_size not in [32, 64, 128, 256]:
        raise ValueError(
            f"Group size {group_size} is not supported for pre-quantized checkpoint."
        )
    if weight_only:
        _replace_linear_with_linear_int4_weight_only_for_pre_quantization(
            module,
            checkpoint,
            group_size,
            dtype,
            dtype,
        )
    else:
        _replace_linear_with_linear_8da4w_for_pre_quantization(
            module,
            checkpoint,
            group_size,
            dtype,
            dtype,
        )
    return module


def _replace_output_linear_with_linear_int8_for_pre_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    dtype: torch.dtype,
):
    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        scales_key = f"{cur_fqn}.scales"
        if (
            isinstance(child, nn.Linear)
            and scales_key in checkpoint
            and "output" in cur_fqn
        ):
            assert checkpoint[f"{cur_fqn}.weight"].dtype == torch.int8
            assert checkpoint[scales_key].dtype == dtype
            return True
        return False

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = Int8DynActInt8WeightLinear(
            device=child.weight.device,
            # pyre-fixme[6]: For 2nd argument expected `int` but got `Union[Module,
            #  Tensor]`.
            in_features=child.in_features,
            # pyre-fixme[6]: For 3rd argument expected `int` but got `Union[Module,
            #  Tensor]`.
            out_features=child.out_features,
            precision=dtype,
            bias=False,
        )
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def transform_output_linear_for_pre_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """
    Transform the model to be able to load pre-quantized checkpoints that
    has the output layer quantized per-channel.
    """
    _replace_output_linear_with_linear_int8_for_pre_quantization(
        module,
        checkpoint,
        dtype,
    )
    return module


def _replace_embedding_with_quantized_group_embedding_for_pre_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    dtype: torch.dtype,
    bit_width: int,
    group_size: Optional[int] = None,
):
    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        # Only replace embedding layers where the checkpoint contains explicit scales
        scales_key = f"{cur_fqn}.scales"
        if isinstance(child, nn.Embedding) and scales_key in checkpoint:
            assert checkpoint[f"{cur_fqn}.weight"].dtype == torch.int8
            assert checkpoint[scales_key].dtype == torch.float32
            return True
        return False

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_embedding = QuantizedGroupEmbedding(
            device=child.weight.device,
            vocab_size=child.weight.shape[0],
            embedding_dim=child.weight.shape[1],
            group_size=group_size,
            dtype=dtype,
            packed=False,  # TODO(lunwenh): support packed embedding for pre-quantized
        )
        return new_embedding

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def transform_embedding_for_pre_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    dtype: torch.dtype,
    bit_width: int,
    group_size: Optional[int] = None,
) -> torch.nn.Module:
    """
    Transform the model to be able to load pre-quantized checkpoints that
    are quantized with the given bit_width and group size for embedding.
    """
    if group_size is not None and group_size not in [0, 32, 64, 128, 256]:
        raise ValueError(
            f"Group size {group_size} is not supported for pre-quantized checkpoint."
        )
    _replace_embedding_with_quantized_group_embedding_for_pre_quantization(
        module,
        checkpoint,
        dtype,
        bit_width,
        group_size,
    )
    return module


def sanitize_checkpoint_from_pre_quantization(
    checkpoint: Any,
):
    """
    Sanitize the pre-quantized checkpoint.
        - Converts all tensors to contiguous format
        - Squeeze all tensors
    """
    for k, v in checkpoint.items():
        checkpoint[k] = torch.squeeze(v.contiguous())
