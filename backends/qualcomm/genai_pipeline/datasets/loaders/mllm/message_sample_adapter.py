# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MLLM message sample dataset loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageSampleAdapter:
    """Loads a multimodal dataset from message-sample JSON files.

    Purpose-agnostic: it produces a component-keyed dataset dict
    (ARTIFACT_TEXT_DECODER, ARTIFACT_AUDIO_ENCODER, ARTIFACT_VISION_ENCODER) and knows nothing about
    calibration / training / eval. The sample paths and model config are fixed
    at construction because encoder dataset construction depends on the model's
    modality config, while ``load_dataset`` only receives call-time shaping
    options.
    """

    def __init__(self, samples_paths: List[str], llm_config: Any) -> None:
        if llm_config is None:
            raise ValueError("llm_config is required for multimodal message samples")
        self._samples_paths = samples_paths
        self._llm_config = llm_config

    def load_dataset(
        self,
        tokenizer: Any,
        max_context_len: int,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load the dataset as ``{component: Dataset}`` for all modalities.

        Args:
            tokenizer: TokenizerWrapper instance.
            max_context_len: Sequence length the decoder samples are encoded at.
            extra_options: Reserved for adapter-specific options; unused here.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_AUDIO_ENCODER,
            ARTIFACT_TEXT_DECODER,
            ARTIFACT_VISION_ENCODER,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.dataset.builders import (
            DecoderDatasetBuilder,
            EncoderDatasetBuilder,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.dataset.loaders import (
            load_conversation_samples,
        )

        logger.info(
            "Loading MLLM message sample dataset: paths=%s",
            self._samples_paths,
        )

        samples = load_conversation_samples(self._samples_paths)

        datasets: Dict[str, Any] = {}

        decoder_builder = DecoderDatasetBuilder(
            tokenizer_wrapper=tokenizer,
            max_context_len=max_context_len,
            is_multimodal=True,
        )
        datasets[ARTIFACT_TEXT_DECODER] = decoder_builder.from_conversation(samples)

        encoder_builder = EncoderDatasetBuilder(
            llm_config=self._llm_config,
            tokenizer_wrapper=tokenizer,
        )
        datasets[ARTIFACT_AUDIO_ENCODER] = encoder_builder.from_message_samples(
            samples, ARTIFACT_AUDIO_ENCODER
        )
        datasets[ARTIFACT_VISION_ENCODER] = encoder_builder.from_message_samples(
            samples, ARTIFACT_VISION_ENCODER
        )

        logger.info(
            "Loaded MLLM message sample dataset: " "%s=%d, %s=%s, %s=%s",
            ARTIFACT_TEXT_DECODER,
            len(datasets[ARTIFACT_TEXT_DECODER]),
            ARTIFACT_AUDIO_ENCODER,
            (
                len(datasets[ARTIFACT_AUDIO_ENCODER])
                if datasets[ARTIFACT_AUDIO_ENCODER]
                else None
            ),
            ARTIFACT_VISION_ENCODER,
            (
                len(datasets[ARTIFACT_VISION_ENCODER])
                if datasets[ARTIFACT_VISION_ENCODER]
                else None
            ),
        )
        return datasets
