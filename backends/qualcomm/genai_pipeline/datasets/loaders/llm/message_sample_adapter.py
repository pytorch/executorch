# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLM message sample dataset loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageSampleAdapter:
    """Loads an LLM dataset from message-sample JSON files.

    Purpose-agnostic: it produces a component-keyed dataset dict and knows
    nothing about calibration / training / eval. The sample paths are fixed at
    construction; ``load_dataset`` only reads shaping options.
    """

    def __init__(self, samples_paths: List[str]) -> None:
        self._samples_paths = samples_paths

    def load_dataset(
        self,
        tokenizer: Any,
        max_context_len: int,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load the dataset as ``{ARTIFACT_TEXT_DECODER: Dataset}``.

        Args:
            tokenizer: TokenizerWrapper instance.
            max_context_len: Sequence length the samples are encoded at.
            extra_options: Reserved for adapter-specific options; unused here.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.dataset.builders import (
            DecoderDatasetBuilder,
        )
        from executorch.examples.qualcomm.oss_scripts.llama.dataset.loaders import (
            load_conversation_samples,
        )

        logger.info(
            "Loading LLM message sample dataset: paths=%s",
            self._samples_paths,
        )

        samples = load_conversation_samples(self._samples_paths)

        decoder_builder = DecoderDatasetBuilder(
            tokenizer_wrapper=tokenizer,
            max_context_len=max_context_len,
            is_multimodal=False,
        )
        dataset = decoder_builder.from_conversation(samples)

        logger.info("Loaded LLM message sample dataset with %d samples", len(dataset))
        return {ARTIFACT_TEXT_DECODER: dataset}
