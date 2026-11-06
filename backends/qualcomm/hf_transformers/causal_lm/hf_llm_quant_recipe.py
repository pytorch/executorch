# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from typing import Optional

import torch
from executorch.backends.qualcomm.quantizer.custom_annotation import annotate_kv_8bit
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from torchao.quantization.pt2e import MinMaxObserver


class LLMQuantRecipe(ABC):
    """
    LLM quantization recipe.

    Subclasses must define ``default_quant_dtype`` and populate ``self.recipe``.
    """

    default_quant_dtype: QuantDtype = None

    def __init__(self):
        self.recipe: Optional[QuantRecipe] = None

        if self.default_quant_dtype is None:
            raise ValueError("default_quant_dtype must be defined in the recipe.")

    def get_kv_io_bit_width(self) -> int:
        if self.default_quant_dtype is None:
            return 32
        elif self.default_quant_dtype == QuantDtype.use_8a8w or annotate_kv_8bit in (
            getattr(c, "func", c) for c in self.recipe.custom_quant_annotations
        ):
            return 8
        else:
            # If quantized but not 8a8w or mix_quantization, it has to be 16bit kv io.
            return 16

    def get_logits_output_bit_width(self) -> int:
        # We use 16bit logits for all quant config
        return 32 if self.default_quant_dtype is None else 16


class Llama3_2_1B_HFQuantRecipe(LLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="default with 16bit activation",
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
                note="Annotate with 16a4w block quantization since these layers are not sensitive.",
            )
            .add_regex(
                {
                    r"model\.lm_head\.conv",
                    r"model\.layers\.[0-3]\.mlp\.down_proj\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
                note="Head and early down_proj are sensitive and should be annotated with 16a8w.",
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class Qwen2_5_0_5B_HFQuantRecipe(LLMQuantRecipe):
    """Ported from Qwen2_5_0_5BQuantRecipe (static_llm_quant_recipe.py:545)."""

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="default with 16bit activation",
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
                note="Annotate with 16a4w block quantization since these layers are not sensitive.",
            )
            .add_regex(
                {r"model\.lm_head\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
                note="Head is sensitive and should be annotated with 16a8w.",
            )
        )


class Qwen2_5_1_5B_HFQuantRecipe(LLMQuantRecipe):
    """Ported from Qwen2_5_1_5BQuantRecipe (static_llm_quant_recipe.py:569).

    static_llama's ``output\\.conv`` head is ``model.lm_head.conv`` in HF.
    """

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="default with 16bit activation",
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 16, 1, 1)},
                note="Annotate with 16a4w block quantization since these layers are not sensitive.",
            )
            .add_regex(
                {r"model\.lm_head\.conv"},
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
                note="Head is sensitive and should be annotated with 16a8w.",
            )
        )


class Qwen3_0_6B_HFQuantRecipe(LLMQuantRecipe):
    """Ported from Qwen3_0_6BQuantRecipe (static_llm_quant_recipe.py:603).

    static_llama's ``layers\\..*\\.feed_forward\\..*w2_conv`` (the down
    projection) is ``model.layers.N.mlp.down_proj.conv`` in HF.
    """

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="default with 16bit activation",
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 32, 1, 1)},
                note="Annotate with 16a4w block quantization since these layers are not sensitive.",
            )
            .add_regex(
                {
                    r"model\.layers\..*\.mlp\.down_proj\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
                note="down_proj is sensitive and should be annotated with 16a8w.",
            )
        )


class Smollm2_HFQuantRecipe(LLMQuantRecipe):
    """Ported from Smollm2QuantRecipe (static_llm_quant_recipe.py:676)."""

    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
            note="default with 16bit activation and 8bit weight",
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
            note="Small model, so keep all conv2d at 16a8w per-channel instead of dropping to 4bit weights.",
        )


class Granite_3_3_2B_Instruct_HFQuantRecipe(LLMQuantRecipe):
    """Ported from Granite_3_3_2B_InstructQuantRecipe (static_llm_quant_recipe.py:403).

    static_llama's ``layers\\..*\\.attention\\..*wv.*`` (the value projection) is
    ``model.layers.N.self_attn.v_proj.conv`` in HF.
    """

    default_quant_dtype = QuantDtype.use_16a4w

    def __init__(self, verbose: bool = False):
        super().__init__()

        self.recipe = (
            QuantRecipe(
                self.default_quant_dtype,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=verbose,
                note="default with 16bit activation",
            )
            .add_node_target(
                {
                    torch.ops.aten.conv2d.default,
                },
                QuantDtype.use_16a4w_block,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, 64, 1, 1)},
                note="Annotate with 16a4w block quantization since these layers are not sensitive.",
            )
            .add_regex(
                {
                    r"model\.layers\..*\.self_attn\.v_proj\.conv",
                },
                QuantDtype.use_16a8w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_CHANNEL,
                note="v_proj is sensitive and should be annotated with 16a8w.",
            )
        )
        self.recipe.custom_quant_annotations.append(annotate_kv_8bit)


class DefaultQuantRecipe(LLMQuantRecipe):
    """When quant recipe is not provided, this will be used"""

    default_quant_dtype = QuantDtype.use_16a8w

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.recipe = QuantRecipe(
            self.default_quant_dtype,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_TENSOR,
            verbose=verbose,
            note="default with 16bit activation and 8bit weight",
        ).add_node_target(
            {
                torch.ops.aten.conv2d.default,
            },
            QuantDtype.use_16a8w,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_CHANNEL,
            note="Conservative fallback for models without a tuned recipe: 16a8w per-channel everywhere.",
        )
