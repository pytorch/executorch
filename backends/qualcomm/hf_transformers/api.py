# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Optional

import torch
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

from torch.export.exported_program import ExportedProgram
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from transformers.exporters import ExecutorchQnnConfig

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

KV_FORWARD = "kv_forward"

HUGGING_FACE_QUANT_RECIPES = {
    "NousResearch/Llama-3.2-1B": Llama3_2_1B_HFQuantRecipe,
    "Qwen/Qwen2.5-0.5B": Qwen2_5_0_5B_HFQuantRecipe,
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen2_5_0_5B_HFQuantRecipe,
    "Qwen/Qwen2.5-1.5B-Instruct": Qwen2_5_1_5B_HFQuantRecipe,
    "Qwen/Qwen3-0.6B": Qwen3_0_6B_HFQuantRecipe,
    "HuggingFaceTB/SmolLM2-135M": Smollm2_HFQuantRecipe,
    "ibm-granite/granite-3.3-2b-instruct": Granite_3_3_2B_Instruct_HFQuantRecipe,
}


def quantize_for_qnn(
    exported_program: ExportedProgram,
    hf_config: ExecutorchQnnConfig,
    sample_inputs: dict,
):
    max_seq_len = sample_inputs["atten_mask"].shape[3]
    quantizer = hf_config.pt2e_quantizer
    graph_module = prepare_pt2e(exported_program.module(), quantizer)

    _pt2e_calibrate(
        graph_module=graph_module,
        calibration_dataset=hf_config.calibration_dataset,
        sample_inputs=sample_inputs,
        max_seq_len=max_seq_len,
        ar_len=1,
    )
    qdq_module = convert_pt2e(graph_module)

    qdq_ep = torch.export.export(
        qdq_module,
        args=(),
        kwargs=copy.deepcopy(dict(sample_inputs)),
        strict=True,
    )

    return qdq_ep


def _pt2e_calibrate(  # noqa: C901
    graph_module: torch.fx.GraphModule,
    calibration_dataset: list[list[int]] | list[str],
    sample_inputs: dict,
    max_seq_len: int,
    ar_len: int,
):
    def calibrate_template(
        module: torch.fx.GraphModule,
        prompt: list[int] | str,
        sample_inputs: dict,
        max_seq_len: int,
        ar_len: int,
    ):
        assert all(
            isinstance(x, int) for x in prompt
        ), "Expected calibration dataset to be tokenized."

        def _build_mask(n_past, past_len, context_len):
            mask = torch.full((1, 1, 1, context_len), -65535.0)
            mask[..., :n_past] = 0.0
            mask[..., past_len:] = 0.0
            return mask

        pos = 0

        token_list = prompt
        past_k, past_v = [], []
        for _ in range(len(sample_inputs["past_k"])):
            past_k.append(torch.zeros(sample_inputs["past_k"][0].shape))
            past_v.append(torch.zeros(sample_inputs["past_v"][0].shape))
        past_len = max_seq_len - ar_len
        context_len = max_seq_len
        # The prefix buffer holds at most past_len slots, so we can advance
        # the position at most past_len times (matching the runner, whose
        # seq_len is clamped to context_len).
        max_seq_len = min(max_seq_len, past_len)

        with torch.no_grad():
            while pos < max_seq_len:
                n_past = min(pos, past_len)
                atten_mask = _build_mask(n_past, past_len, context_len)
                input_pos = torch.tensor([[n_past]], dtype=torch.int32)
                logits, new_k, new_v = module(
                    torch.full((1, 1), token_list[pos], dtype=torch.int32),
                    atten_mask,
                    input_pos,
                    past_k,
                    past_v,
                )
                # Prefix append: write the new slot into buffer at slot n_past.
                for layer_idx in range(len(past_k)):
                    past_k[layer_idx][..., :, n_past] = new_k[layer_idx][..., :, 0]
                    past_v[layer_idx][..., n_past, :] = new_v[layer_idx][..., 0, :]
                pos += 1
                if pos >= len(token_list):
                    token_list.append(torch.argmax(logits, dim=-1).item())
        logging.info(f"Token result of LLM with static cache:\n {token_list} \n\n\n")

    for prompt in calibration_dataset:
        calibrate_template(
            module=graph_module,
            prompt=prompt,
            sample_inputs=sample_inputs,
            max_seq_len=max_seq_len,
            ar_len=ar_len,
        )

    logging.info("Calibration finish...")


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
