# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLM HuggingFace chat dataset loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HFDatasetAdapter:
    """Loads an LLM dataset from a HuggingFace chat dataset.

    Purpose-agnostic: it produces a component-keyed dataset dict and knows
    nothing about calibration / training / eval. The dataset name and sample
    count are fixed at construction; ``load_dataset`` only reads shaping
    options.
    """

    def __init__(self, dataset_name: str, limit: int) -> None:
        self._dataset_name = dataset_name
        self._limit = limit

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

        logger.info(
            "Loading LLM HuggingFace dataset: dataset=%s, limit=%s",
            self._dataset_name,
            self._limit,
        )

        decoder_builder = DecoderDatasetBuilder(
            tokenizer_wrapper=tokenizer,
            max_context_len=max_context_len,
            is_multimodal=False,
        )
        dataset = decoder_builder.from_hf_source(
            self._dataset_name,
            num_samples=self._limit,
        )

        logger.info("Loaded LLM HuggingFace dataset with %d samples", len(dataset))
        return {ARTIFACT_TEXT_DECODER: dataset}
