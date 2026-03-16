# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

import logging
from typing import Optional, Tuple, Union
import random

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from executorch.extension.llm.export.builder import LLMEdgeManager
from torchao.quantization.pt2e.quantizer import Quantizer

try:
    import nncf  # type: ignore[import-untyped]
    from pytorch_tokenizers import get_tokenizer  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("Please install nncf via backends/openvino/requirements.txt")

TASK_TO_HF_DATASET = {
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "train",
    },
}


# This code is adapted from https://github.com/pytorch/executorch/blob/0c54fd0483314da173f8e14d63d2ed9591c7133a/extension/llm/export/builder.py#L278
def get_calibration_data(
    tokenizer,
    data: str,
    nsamples: int,
    seqlen: int,
):
    """
    This method is used to obtain calibration data from a prompt so that the algorithm
    is calibrated not only with the dataset but also the inputs which are output by
    the model.
    Currently, this method is only tested with Llama models.
    """
    # Copied from optimum.gptq.data.get_wikitext2 with added computation of `limit` variable:
    limit = nsamples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in data["text"][:limit]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        dataset.extend([(token, pos) for pos, token in enumerate(inp)])
    return dataset


def transform_fn(token_pos_map: Tuple[int, int]):
    """
    Transforms and returns input from dataset so that it is acceptable by the model
    Currently, this method is only tested with Llama models.

    :param token_pos_map: This input contains the position and its token ID
    """
    inputs = (
        torch.tensor([[token_pos_map[0]]]),
        {"input_pos": torch.tensor([token_pos_map[1]])},
    )

    return inputs


def _build_nncf_calibration_dataset(
    calibration_task: Optional[str],
    tokenizer,
    seq_len: Optional[int],
    subset_size: Optional[int],
    awq: bool,
    scale_estimation: bool,
):
    if not (awq or scale_estimation):
        return None

    if subset_size is None or subset_size <= 0:
        raise ValueError("subset_size must be a positive integer when calibration is enabled.")

    has_calibration_inputs = (
        calibration_task is not None and tokenizer is not None and seq_len is not None
    )

    # Scale estimation requires full calibration setup.
    if scale_estimation and not has_calibration_inputs:
        missing_params = []
        if calibration_task is None:
            missing_params.append("calibration_task")
        if tokenizer is None:
            missing_params.append("tokenizer")
        if seq_len is None:
            missing_params.append("seq_len")
        raise ValueError(
            "Missing required calibration parameter(s): "
            + ", ".join(missing_params)
            + ". Please provide calibration_task, tokenizer, and seq_len."
        )

    if not has_calibration_inputs:
        return None

    if calibration_task not in TASK_TO_HF_DATASET:
        raise ValueError(
            f"Unsupported calibration task: {calibration_task}. Supported tasks are: {list(TASK_TO_HF_DATASET.keys())}"
        )

    dataset = load_dataset(**TASK_TO_HF_DATASET[calibration_task])
    calibration_data = get_calibration_data(
        tokenizer,
        dataset,
        subset_size,
        seq_len,
    )

    return nncf.Dataset(
        calibration_data,
        transform_func=transform_fn,
    )


def apply_nncf_data_aware_compression(
    builder_or_model: Union[LLMEdgeManager, torch.fx.GraphModule],
    quantizer: Quantizer,
    awq: bool,
    scale_estimation: bool,
    calibration_task: Optional[str] = "wikitext",
    tokenizer: Optional[str] = None,
    seq_len: Optional[int] = None,
    subset_size: Optional[int] = 1024,
) -> Union[LLMEdgeManager, torch.fx.GraphModule]:
    """
    Applies NNCF data-aware weight compression to the exported LLM graph.
    Uses the builder's tokenizer and calibration prompt to generate token-level
    calibration data, then runs `nncf.experimental.torch.fx.compress_pt2e` with
    the given quantizer and optional AWQ / scale estimation enabled.

    :param builder_or_model: Either:
        - LLMEdgeManager containing the FX graph, tokenizer path,
          calibration prompt, and max sequence length, or
        - torch.fx.GraphModule to be compressed directly.
    :param quantizer: TorchAO quantizer to use for compression.
    :param awq: If True, enables Activation-aware Weights Quantization (AWQ).
    :param scale_estimation: If True, enables NNCF's scale estimation algorithm.
    :param calibration_task: Optional task key for calibration dataset when passing
        GraphModule directly (e.g. "wikitext", "c4", "gsm8k").
    :param tokenizer: Optional tokenizer when passing GraphModule directly.
    :param seq_len: Optional max sequence length of each calibration prompt when passing GraphModule directly.
    :param subset_size: Optional max number of samples from the calibration dataset to use for calibration.
        Default is 1024. This is high because it is token-level data, not sample-level. The number of tokens is much higher than the number of samples.
    :return: Updated input object with compressed torch FX model.
    """
    nncf_calibration_data = None

    if not quantizer:
        logging.info("No quantizer provided, skipping NNCF compression.")
        return builder_or_model

    if isinstance(builder_or_model, LLMEdgeManager):
        builder = builder_or_model
        model = builder.pre_autograd_graph_module
        tokenizer_path = builder.tokenizer_path
        tokenizer = get_tokenizer(tokenizer_path) if tokenizer_path is not None else None
        # Keeping it to model's max length for now, but this can be decoupled in the future if needed
        seq_len = builder.max_seq_len
    elif isinstance(builder_or_model, torch.fx.GraphModule):
        builder = None
        model = builder_or_model
    else:
        raise TypeError(
            "builder_or_model must be either LLMEdgeManager or torch.fx.GraphModule"
        )

    nncf_calibration_data = _build_nncf_calibration_dataset(
        calibration_task=calibration_task,
        tokenizer=tokenizer,
        seq_len=seq_len,
        subset_size=subset_size,
        awq=awq,
        scale_estimation=scale_estimation,
    )

    compressed_model = nncf.experimental.torch.fx.compress_pt2e(
        model,
        quantizer=quantizer,
        dataset=nncf_calibration_data,
        awq=awq,
        scale_estimation=scale_estimation,
        subset_size=subset_size,
    )

    if builder is not None:
        builder.pre_autograd_graph_module = compressed_model
        return builder

    return compressed_model
