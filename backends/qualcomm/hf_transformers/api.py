# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from executorch.backends.qualcomm.hf_transformers.causal_lm.hf_llm_quant_recipe import (
    DefaultQuantRecipe,
    Granite_3_3_2B_Instruct_HFQuantRecipe,
    Llama3_2_1B_HFQuantRecipe,
    LLMQuantRecipe,
    Qwen2_5_0_5B_HFQuantRecipe,
    Qwen2_5_1_5B_HFQuantRecipe,
    Qwen3_0_6B_HFQuantRecipe,
    Smollm2_HFQuantRecipe,
)

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

HUGGING_FACE_QUANT_RECIPES = {
    "NousResearch/Llama-3.2-1B": Llama3_2_1B_HFQuantRecipe,
    "Qwen/Qwen2.5-0.5B": Qwen2_5_0_5B_HFQuantRecipe,
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen2_5_0_5B_HFQuantRecipe,
    "Qwen/Qwen2.5-1.5B-Instruct": Qwen2_5_1_5B_HFQuantRecipe,
    "Qwen/Qwen3-0.6B": Qwen3_0_6B_HFQuantRecipe,
    "HuggingFaceTB/SmolLM2-135M": Smollm2_HFQuantRecipe,
    "ibm-granite/granite-3.3-2b-instruct": Granite_3_3_2B_Instruct_HFQuantRecipe,
}


def get_quant_recipe(model_id, verbose=True):
    """
    model_id = The Hugging Face Model ID.
    verbose: If True, it will the quant config tagged for each node.
    """
    recipe_cls = HUGGING_FACE_QUANT_RECIPES.get(model_id, DefaultQuantRecipe)
    if recipe_cls == DefaultQuantRecipe:
        logging.warning(
            f"{model_id} does not have customized quant recipe using default quant recipe."
        )
    quant_recipe: Optional[LLMQuantRecipe] = recipe_cls(verbose) if recipe_cls else None
    return quant_recipe
