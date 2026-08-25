# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver


class EncoderQuantRecipe:
    """
    Qualcomm's Encoder quantization recipe.
    """

    def __init__(self):
        self.recipe: Optional[QuantRecipe] = None

        self.default_quant_dtype = getattr(self, "default_quant_dtype", None)
        if self.default_quant_dtype is None:
            raise ValueError("default_quant_dtype must be defined in the recipe.")

    def annotate(self, graph_module: torch.fx.GraphModule):
        self.recipe.annotate(graph_module)

    def get_act_bit_width(self) -> int:
        return 32 if self.default_quant_dtype is None else 16


class InternVL3EncoderQuantRecipe(EncoderQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.linear.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class SmolVLMEncoderQuantRecipe(EncoderQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
        ).add_node_target(
            {
                torch.ops.aten.linear.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
        )


class GraniteSpeechEncoderQuantRecipe(EncoderQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=HistogramObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
            )
            .add_node_target(
                {
                    torch.ops.aten.linear.default,
                    torch.ops.aten.conv1d.default,
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=HistogramObserver,
                granularity=QuantGranularity.PER_CHANNEL,
            )
            .add_regex(
                {
                    r"encoder\..*\.layers\..*\.conv\.up_conv",
                    r"encoder\..*\.layers\..*\.conv\.down_conv",
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=HistogramObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1)},
            )
        )
