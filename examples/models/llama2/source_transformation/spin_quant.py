# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Helper functions for tranforming the model to be able to run SpinQuant.
# See https://github.com/facebookresearch/SpinQuant for more details about SpinQuant.

from typing import Any

import torch

import torch.nn.functional as F

from executorch.examples.models.llama2.llama_transformer import FeedForward
from torch import nn
from torchao.quantization.GPTQ import _check_linear_int4_k, Int8DynActInt4WeightLinear
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter


def _inject_fast_hadamard_transform_cuda_for_spin_quant(module: torch.nn.Module):
    """
    SpinQuant needs two Hadmard matrixes: R3 and R4. Here we are only injecting R4 in the feed forward layer.
    R3 needs to be injected as well when KV cache quantization is enabled.
    """
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        raise ImportError(
            "Please install fast-hadamard-transform: pip install fast-hadamard-transform"
        )

    class FeedForwardCustom(nn.Module):
        def __init__(self, w1, w2, w3):
            super().__init__()
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3

        def forward(self, x):
            w = F.silu(self.w1(x)) * self.w3(x)
            n = w.shape[-1]
            return self.w2(hadamard_transform(w.contiguous()) / torch.tensor(n).sqrt())

    for name, child in module.named_children():
        if isinstance(child, FeedForward):
            setattr(module, name, FeedForwardCustom(child.w1, child.w2, child.w3))
        else:
            _inject_fast_hadamard_transform_cuda_for_spin_quant(child)


def inject_fast_hadamard_transform_cuda_for_spin_quant(
    module: torch.nn.Module,
) -> torch.nn.Module:
    _inject_fast_hadamard_transform_cuda_for_spin_quant(module)
    return module


def _replace_linear_with_linear_8da4w_for_spin_quant(
    module: torch.nn.Module,
    checkpoint: Any,
    group_size: int,
    precision: torch.dtype,
    scales_precision: torch.dtype,
):
    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        # Only replace linear layers where the checkpoint contains explicit scales
        scales_key = f"{cur_fqn}.scale"
        if isinstance(child, nn.Linear) and scales_key in checkpoint:
            assert _check_linear_int4_k(child.in_features, group_size)
            assert checkpoint[f"{cur_fqn}.weight"].dtype == torch.int8
            assert checkpoint[scales_key].dtype == scales_precision
            return True
        return False

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = Int8DynActInt4WeightLinear(
            child.in_features,
            child.out_features,
            bias=False,
            device=child.weight.device,
            groupsize=group_size,
            precision=precision,
            scales_precision=scales_precision,
        )
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def transform_for_spinquant(
    module: torch.nn.Module,
    checkpoint: Any,
    group_size: int,
    quantization_mode: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """
    Transform the model to be able to load SpinQuant checkpoints that
    are quantized with the given group size and quantization mode.
    """

    if group_size not in [32, 64, 128, 256]:
        raise ValueError(f"Group size {group_size} is not supported for SpinQuant.")
    if quantization_mode not in ["8da4w"]:
        raise ValueError(
            f"Quantization mode {quantization_mode} is not compatible with SpinQuant."
        )
    _replace_linear_with_linear_8da4w_for_spin_quant(
        module,
        checkpoint,
        group_size,
        dtype,
        dtype,
    )
    return module


def sanitize_checkpoint_from_spinquant(
    checkpoint: Any,
    group_size: int,
):
    """
    Sanitize the SpinQuant checkpoint.
        - Renames 'scale' to 'scales'
        - Groups scales
        - Removes 'o_weight'
        - Converts all tensors to contiguous format
    """
    keys_to_rename = []
    keys_to_remove = []
    for k, _ in checkpoint.items():
        if k.endswith(".scale"):
            new_key = k + "s"
            keys_to_rename.append((k, new_key))
        if k.endswith(".o_weight"):
            keys_to_remove.append(k)

    for old_key, new_key in keys_to_rename:
        old_val = checkpoint.pop(old_key)
        checkpoint[new_key] = old_val if group_size == -1 else old_val[:, ::group_size]
    for k in keys_to_remove:
        checkpoint.pop(k)
    for k, v in checkpoint.items():
        checkpoint[k] = v.contiguous()
