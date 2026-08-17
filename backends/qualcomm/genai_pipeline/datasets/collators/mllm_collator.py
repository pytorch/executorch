# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLLM component-aware collator provider."""

from __future__ import annotations

from typing import Any, Dict


class MLLMDatasetCollector:
    """Provides the collator for each MLLM component.

    ARTIFACT_TEXT_DECODER uses ``LLMCalibCollator``; encoder components
    (ARTIFACT_VISION_ENCODER / ARTIFACT_AUDIO_ENCODER) use ``ModalityEncoderCollator``. Building
    the DataLoaders is the purpose adapter's responsibility; this collector
    only maps each component to its collate function.
    """

    def create_collators(
        self,
        example_inputs: Dict[str, Any],
        max_context_len: int,
    ) -> Dict[str, Any]:
        import torch
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_AUDIO_ENCODER,
            ARTIFACT_TEXT_DECODER,
            ARTIFACT_VISION_ENCODER,
        )

        from executorch.examples.qualcomm.oss_scripts.llama.dataset.collators import (
            LLMCalibCollator,
            ModalityEncoderCollator,
        )

        if (
            not isinstance(example_inputs, dict)
            or ARTIFACT_TEXT_DECODER not in example_inputs
        ):
            raise ValueError(
                f"MLLM example_inputs must be a dict containing {ARTIFACT_TEXT_DECODER}; "
                f"got {type(example_inputs)}"
            )

        decoder_inputs = example_inputs[ARTIFACT_TEXT_DECODER]
        if not isinstance(decoder_inputs, tuple) or len(decoder_inputs) < 2:
            raise ValueError(
                f"{ARTIFACT_TEXT_DECODER} example_inputs must be a tuple of "
                f"(tokens, attention_mask, ...); got {type(decoder_inputs)}"
            )

        attn_mask = decoder_inputs[1]
        if attn_mask is None:
            raise ValueError(
                f"attention_mask not found at {ARTIFACT_TEXT_DECODER} example_inputs[1]"
            )

        encoder_collator = ModalityEncoderCollator()
        return {
            ARTIFACT_TEXT_DECODER: LLMCalibCollator(
                attn_mask_template=attn_mask,
                max_context_len=max_context_len,
                token_dtype=torch.int32,
            ),
            ARTIFACT_VISION_ENCODER: encoder_collator,
            ARTIFACT_AUDIO_ENCODER: encoder_collator,
        }
