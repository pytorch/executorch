# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python bindings for ExecuTorch MultimodalRunner.

This module provides a Python interface to the ExecuTorch multimodal LLM runner,
enabling processing of mixed inputs (text, images, audio) and text generation.
"""

try:
    # Import shared components from the compiled C++ extension
    from executorch.extension.llm.runner._llm_runner import (  # noqa: F401
        GenerationConfig,
        Image,
        make_audio_input,
        make_image_input,
        make_raw_audio_input,
        make_text_input,
        make_token_input,
        MultimodalInput,
        MultimodalRunner,
        Stats,
    )
except ImportError:
    raise RuntimeError(
        "LLM runner is not installed. Please build ExecuTorch from source with EXECUTORCH_BUILD_PYBIND=ON"
    )


import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch


def _find_image_token_runs(
    input_ids: torch.Tensor, image_token_id: Optional[int]
) -> List[tuple[int, int, int]]:
    """Return contiguous runs (start, end, length) of image_token_id in input_ids.

    input_ids must be a 1D torch.Tensor. If image_token_id is None, returns an empty list.
    """
    if image_token_id is None:
        return []

    ids_list = input_ids.tolist()
    runs: List[tuple[int, int, int]] = []
    i = 0
    L = len(ids_list)
    while i < L:
        if ids_list[i] == image_token_id:
            j = i
            while j < L and ids_list[j] == image_token_id:
                j += 1
            runs.append((i, j - 1, j - i))
            i = j
        else:
            i += 1

    return runs


def _hf_to_multimodal_inputs(  # noqa: C901
    inputs: Dict[str, Any], image_token_id: Optional[int] = None
) -> List[MultimodalInput]:
    """Convert a HuggingFace AutoProcessor dict to ExecuTorch MultimodalInputs.
    Currently only support 1 image inside the input.

    Args:
      - inputs: A dictionary containing the input data.
      - image_token_id: The token ID for the image, if present.

    `inputs` expected keys:
      - 'input_ids': torch.Tensor of shape (L,) or (1, L)
      - Optional 'pixel_values': torch.Tensor; if present, must also provide
        'image_token_id' (or alias 'image_token_index') and there must be
        exactly one image token occurrence in input_ids.

    Raises:
      RuntimeError: missing keys, invalid shapes/dtypes, or unsupported cases.
    """
    if "input_ids" not in inputs:
        raise RuntimeError("HF inputs dict must contain 'input_ids' (torch.Tensor)")

    input_ids = inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("'input_ids' must be a torch.Tensor")

    if input_ids.dim() == 2:
        if input_ids.size(0) != 1:
            raise RuntimeError(
                "Expected 'input_ids' with batch size 1 when 2D (shape (1, L))"
            )
        input_ids = input_ids.squeeze(0)
    if input_ids.dim() != 1:
        raise RuntimeError("'input_ids' must be 1D (L) or 2D with batch size 1")

    has_pixel_values = "pixel_values" in inputs

    # If pixel_values in dict, require image_token_id
    if has_pixel_values and image_token_id is None:
        raise RuntimeError("'pixel_values' provided but missing 'image_token_id'")

    # If there are image token ids but no pixel_values, it's an error
    if (
        image_token_id is not None
        and (input_ids == image_token_id).any().item()
        and not has_pixel_values
    ):
        raise RuntimeError(
            "Found image token(s) in input_ids but 'pixel_values' not provided"
        )

    # No images: return a single tokens input
    if not has_pixel_values:
        return [make_token_input(input_ids.to(torch.long).tolist())]

    # Determine number of images from pixel_values shape
    pv = inputs["pixel_values"]
    if not isinstance(pv, torch.Tensor):
        raise RuntimeError(
            "'pixel_values' must be a torch.Tensor, run with `return_tensors='pt'` in HF processor"
        )
    if pv.dim() == 4:
        num_images = int(pv.size(0))
    elif pv.dim() == 3:
        num_images = 1
    else:
        raise RuntimeError(
            f"'pixel_values' must be 3D (C,H,W) or 4D (N,C,H,W)/(N,H,W,C), got shape {pv.shape}"
        )

    # Only support batch size 1 for now:
    if num_images != 1:
        raise RuntimeError("Only 1 image is supported for now")
    # Find contiguous runs of image_token_id in input_ids
    runs = _find_image_token_runs(input_ids, image_token_id)

    if len(runs) == 0:
        raise RuntimeError(
            "'pixel_values' provided but no occurrence of 'image_token_id' in input_ids"
        )

    # Support only one image/run for now; enforce exact match
    if num_images != 1 or len(runs) != 1:
        raise RuntimeError(
            f"Mismatch between images and image token runs: images={num_images}, runs={len(runs)} (only batch=1 and a single contiguous run are supported)"
        )

    first, last, _ = runs[0]

    combined: List[MultimodalInput] = []
    if first > 0:
        combined.append(make_token_input(input_ids[:first].to(torch.long).tolist()))

    # Use C++ checked creator for images (handles 3D/4D, CHW/HWC, uint8/float32)
    combined.append(make_image_input(inputs["pixel_values"]))

    if (last + 1) < input_ids.numel():
        combined.append(make_token_input(input_ids[last + 1 :].to(torch.long).tolist()))

    return combined


def generate_hf(
    runner: MultimodalRunner,
    inputs: Union[Dict[str, Any], List[MultimodalInput]],
    config: GenerationConfig,
    image_token_id: Optional[int] = None,
    token_callback: Optional[Callable[[str], None]] = None,
    stats_callback: Optional[Callable[[Stats], None]] = None,
) -> None:
    """Generate using an HF dict by converting to multimodal inputs internally, or using a list of MultimodalInput."""
    if isinstance(inputs, dict):
        logging.info(
            "Input is a dict, assuming it's coming from HF AutoProcessor.apply_chat_template(). Converting to multimodal inputs."
        )
        converted = _hf_to_multimodal_inputs(inputs, image_token_id=image_token_id)
    else:
        converted = inputs

    runner.generate(converted, config, token_callback, stats_callback)


def generate_text_hf(
    runner: MultimodalRunner,
    inputs: Union[Dict[str, Any], List[MultimodalInput]],
    config: GenerationConfig,
    image_token_id: Optional[int] = None,
) -> str:
    """Generate using an HF dict by converting to multimodal inputs internally, or using a list of MultimodalInput."""
    if isinstance(inputs, dict):
        logging.info(
            "Input is a dict, assuming it's coming from HF AutoProcessor.apply_chat_template(). Converting to multimodal inputs."
        )
        converted = _hf_to_multimodal_inputs(inputs, image_token_id=image_token_id)
    else:
        converted = inputs

    return runner.generate_text(converted, config)


setattr(MultimodalRunner, "generate_hf", generate_hf)  # noqa B010
setattr(MultimodalRunner, "generate_text_hf", generate_text_hf)  # noqa B010


__all__ = [
    "GenerationConfig",
    "Image",
    "make_audio_input",
    "make_image_input",
    "make_raw_audio_input",
    "make_text_input",
    "make_token_input",
    "MultimodalInput",
    "MultimodalRunner",
    "Stats",
]
