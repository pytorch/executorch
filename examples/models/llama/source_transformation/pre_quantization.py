# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Helper functions for tranforming the model to be able to load pre-quantized checkpoints.

from typing import Any, Optional

import torch
from torch import nn

from torchao.quantization.GPTQ import _check_linear_int4_k, Int8DynActInt4WeightLinear
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter

from .quantize import Int8DynActInt8WeightLinear, QuantizedGroupEmbedding


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
            bias=False,
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
) -> torch.nn.Module:
    """
    Transform the model to be able to load pre-quantized checkpoints that
    are quantized with the given group size and quantization mode for
    linear layers.
    """

    if group_size not in [32, 64, 128, 256]:
        raise ValueError(
            f"Group size {group_size} is not supported for pre-quantized checkpoint."
        )
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
