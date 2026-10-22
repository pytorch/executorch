# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GenAI Pipeline quantization recipes.

Re-exports quantization recipe classes from the existing llama module,
providing a stable import path within the genai_pipeline namespace.

Usage:
    from backends.qualcomm.genai_pipeline.recipes import (
        StaticLLMQuantRecipe,
        Llama3_1BQuantRecipe,
    )
"""

from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    CodegenQuantRecipe,
    Gemma2QuantRecipe,
    Gemma3QuantRecipe,
    Gemma_2BQuantRecipe,
    GLM_1_5B_InstructQuantRecipe,
    Granite_3_3_2B_InstructQuantRecipe,
    GraniteSpeech_3_3_2B_InstructQuantRecipe,
    InternVL3_1B_QuantRecipe,
    Llama3_1BQuantRecipe,
    Llama3_3BQuantRecipe,
    LlamaStories110MQuantRecipe,
    LlamaStories260KQuantRecipe,
    Phi4MiniQuantRecipe,
    Qwen2_5_0_5BQuantRecipe,
    Qwen2_5_1_5BQuantRecipe,
    Qwen3_0_6BQuantRecipe,
    Qwen3_1_7BQuantRecipe,
    Smollm2QuantRecipe,
    Smollm3QuantRecipe,
    SmolVLMQuantRecipe,
    StaticLLMQuantRecipe,
)

__all__ = [
    "StaticLLMQuantRecipe",
    "CodegenQuantRecipe",
    "Gemma2QuantRecipe",
    "Gemma3QuantRecipe",
    "Gemma_2BQuantRecipe",
    "GLM_1_5B_InstructQuantRecipe",
    "Granite_3_3_2B_InstructQuantRecipe",
    "GraniteSpeech_3_3_2B_InstructQuantRecipe",
    "InternVL3_1B_QuantRecipe",
    "Llama3_1BQuantRecipe",
    "Llama3_3BQuantRecipe",
    "LlamaStories110MQuantRecipe",
    "LlamaStories260KQuantRecipe",
    "Phi4MiniQuantRecipe",
    "Qwen2_5_0_5BQuantRecipe",
    "Qwen2_5_1_5BQuantRecipe",
    "Qwen3_0_6BQuantRecipe",
    "Qwen3_1_7BQuantRecipe",
    "Smollm2QuantRecipe",
    "Smollm3QuantRecipe",
    "SmolVLMQuantRecipe",
]
