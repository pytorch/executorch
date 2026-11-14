"""Concrete QNN quantization recipes for SmolVLA components."""

from __future__ import annotations

import torch
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from torchao.quantization.pt2e import MinMaxObserver


class SmolVLAPrefixQuantRecipe:
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        )
        self.recipe.add_node_target(
            {torch.ops.aten.linear.default},
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
            note="SmolVLA linear weights use per-channel quantization.",
        )


class SmolVLAActionExpertQuantRecipe:
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        )
        self.recipe.add_node_target(
            {torch.ops.aten.linear.default},
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
            note="SmolVLA linear weights use per-channel quantization.",
        )


__all__ = [
    "SmolVLAActionExpertQuantRecipe",
    "SmolVLAPrefixQuantRecipe",
]
