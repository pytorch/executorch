# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Helper functions for tranforming the model to be able to load checkpoints with
# LoRA adaptors. See https://arxiv.org/abs/2106.09685 for more details about LoRA.

from typing import Any

import torch
from torch import nn
from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter


class LoRAAdaptorLinear(nn.Module):
    """
    LoRA adaptor for linear layers.

    This class implements Low-Rank Adaptation(LoRA) for linear layers.
    See more details about LoRA here https://arxiv.org/abs/2106.09685.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scale: float = 2.0,
        dtype=torch.float32,
        device=None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.A = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        self.B = nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.B(self.A(x))  # pyre-ignore[7]


class Int8DynActInt4WeightLinearLoRA(Int8DynActInt4WeightLinear):
    """
    Int8DynActInt4WeightLinear with LoRA adaptor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        bias=True,
        device=None,
        groupsize: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
        lora_adaptor_precision: torch.dtype = torch.bfloat16,
        lora_scale: float = 2.0,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            groupsize=groupsize,
            precision=precision,
            scales_precision=scales_precision,
        )
        # TODO(lunwenh): Remove this once TorchAO's commit pin in ExecuTorch is updated to include this PR
        self.zeros = torch.zeros_like(self.zeros)
        self.adaptor = LoRAAdaptorLinear(
            in_features,
            out_features,
            lora_rank,
            scale=lora_scale,
            dtype=lora_adaptor_precision,
            device=device,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input) + self.adaptor(input).to(dtype=self.precision)


def _replace_linear_8da4w_for_lora(
    module: torch.nn.Module,
    checkpoint: Any,
    lora_rank: int,
):
    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        # Only replace linear layers where the checkpoint contains explicit adaptors
        adaptor_A_key = f"{cur_fqn}.adaptor.A.weight"
        adaptor_B_key = f"{cur_fqn}.adaptor.B.weight"
        if (
            isinstance(child, Int8DynActInt4WeightLinear)
            and adaptor_A_key in checkpoint
            and adaptor_B_key in checkpoint
        ):
            assert checkpoint[adaptor_A_key].dtype == torch.bfloat16
            assert checkpoint[adaptor_A_key].shape[0] == lora_rank
            assert checkpoint[adaptor_A_key].shape[1] == child.in_features
            assert checkpoint[adaptor_B_key].dtype == torch.bfloat16
            assert checkpoint[adaptor_B_key].shape[0] == child.out_features
            assert checkpoint[adaptor_B_key].shape[1] == lora_rank
            return True
        return False

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        new_linear = Int8DynActInt4WeightLinearLoRA(
            # pyre-fixme[6]: For 1st argument expected `int` but got `Union[Module,
            #  Tensor]`.
            child.in_features,
            # pyre-fixme[6]: For 2nd argument expected `int` but got `Union[Module,
            #  Tensor]`.
            child.out_features,
            lora_rank=lora_rank,
            bias=False,
            device=child.weight.device,
            # pyre-fixme[6]: For 6th argument expected `int` but got `Union[Module,
            #  Tensor]`.
            groupsize=child.groupsize,
            # pyre-fixme[6]: For 7th argument expected `dtype` but got
            #  `Union[Module, Tensor]`.
            precision=child.precision,
            # pyre-fixme[6]: For 8th argument expected `dtype` but got
            #  `Union[Module, dtype, Tensor]`.
            scales_precision=child.scales.dtype,
        )
        return new_linear

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def transform_linear_for_lora_after_quantization(
    module: torch.nn.Module,
    checkpoint: Any,
    lora_rank: int,
) -> torch.nn.Module:
    """
    Transform the model to be able to load checkpoints with LoRA adaptors.
    The model should be already transformed to be able to load pre-quantized
    checkpoints. The checkpoint should have been pre-quantized and added with
    LoRA adaptors.
    """
    _replace_linear_8da4w_for_lora(
        module,
        checkpoint,
        lora_rank,
    )
    return module
