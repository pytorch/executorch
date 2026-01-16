# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from typing import Callable, List, Optional, Tuple, Union

import torch
from executorch.extension.llm.export.builder import LLMEdgeManager
from torchao.quantization.pt2e.quantizer import Quantizer

try:
    import nncf  # type: ignore[import-untyped]
    from pytorch_tokenizers import get_tokenizer  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("Please install nncf via backends/openvino/requirements.txt")


# This code is taken from https://github.com/pytorch/executorch/blob/0c54fd0483314da173f8e14d63d2ed9591c7133a/extension/llm/export/builder.py#L278
def get_calibration_data(
    module: torch.fx.GraphModule, tokenizer, prompts: str, max_len: int
):
    """
    This method is used to obtain calibration data from a prompt so that the algorithm
    is calibrated not only with the dataset but also the inputs which are output by
    the model.
    Currently, this method is only tested with Llama models.
    """
    # TODO: change criteria & support batch inputs if necessary
    pos = torch.tensor(0, dtype=torch.int64)
    token_list = tokenizer.encode(prompts, bos=True, eos=False)

    with torch.no_grad():
        while token_list[-1] != tokenizer.eos_id and pos < max_len:
            logits = module(
                torch.full((1, 1), token_list[pos]),
                {"input_pos": torch.tensor((pos,))},
            )
            pos += 1
            if pos >= len(token_list):
                token_list.append(torch.argmax(logits[:], dim=-1).item())
    token_list = [
        (
            pos,
            token,
        )
        for pos, token in enumerate(token_list)
    ]
    return token_list


def transform_fn(token_pos_map: Tuple[int, int]):
    """
    Transforms and returns input from dataset so that it is acceptable by the model
    Currently, this method is only tested with Llama models.

    :param token_pos_map: This input contains the position and its token ID
    """
    inputs = (
        torch.tensor(token_pos_map[1]).unsqueeze(0).unsqueeze(0),
        {"input_pos": torch.tensor([token_pos_map[0]])},
    )

    return inputs


def apply_nncf_data_aware_compression(
    builder_exported: LLMEdgeManager,
    quantizer: Quantizer,
    awq: bool,
    scale_estimation: bool,
) -> LLMEdgeManager:
    """
    Applies NNCF data-aware weight compression to the exported LLM graph.
    Uses the builder's tokenizer and calibration prompt to generate token-level
    calibration data, then runs `nncf.experimental.torch.fx.compress_pt2e` with
    the given quantizer and optional AWQ / scale estimation enabled.

    :param builder_exported: LLMEdgeManager containing the FX graph, tokenizer path,
        calibration prompt, and max sequence length.
    :param quantizer: TorchAO quantizer to use for compression.
    :param awq: If True, enables Activation-aware Weights Quantization (AWQ).
    :param scale_estimation: If True, enables NNCF's scale estimation algorithm.
    :return: The updated LLMEdgeManager with compressed torch FX model
    """
    tokenizer = get_tokenizer(builder_exported.tokenizer_path)

    nncf_calibration_data = None
    if awq or scale_estimation:
        nncf_calibration_data = nncf.Dataset(
            get_calibration_data(
                builder_exported.pre_autograd_graph_module,
                tokenizer,
                builder_exported.calibration_data,
                builder_exported.max_seq_len,
            ),
            transform_func=transform_fn,
        )

    builder_exported.pre_autograd_graph_module = (
        nncf.experimental.torch.fx.compress_pt2e(
            builder_exported.pre_autograd_graph_module,
            quantizer=quantizer,
            dataset=nncf_calibration_data,
            awq=awq,
            scale_estimation=scale_estimation,
        )
    )
    return builder_exported
