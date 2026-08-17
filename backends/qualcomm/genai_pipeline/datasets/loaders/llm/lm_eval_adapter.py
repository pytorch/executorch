# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLM lm_eval dataset loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class LMEvalAdapter:
    """Loads an LLM dataset from lm_eval tasks.

    Purpose-agnostic: it produces a component-keyed dataset dict and knows
    nothing about calibration / training / eval. The task selection is fixed at
    construction; ``load_dataset`` only reads shaping options.
    """

    def __init__(
        self,
        tasks: Union[str, List[str]],
        limit: int,
        num_fewshot: Optional[int] = None,
    ) -> None:
        self._tasks = tasks
        self._limit = limit
        self._num_fewshot = num_fewshot

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
            "Loading LLM lm_eval dataset: tasks=%s, limit=%s",
            self._tasks,
            self._limit,
        )

        decoder_builder = DecoderDatasetBuilder(
            tokenizer_wrapper=tokenizer,
            max_context_len=max_context_len,
            is_multimodal=False,
        )
        dataset = decoder_builder.from_lm_eval(
            tasks=self._tasks,
            limit=self._limit,
            num_fewshot=self._num_fewshot,
        )

        logger.info("Loaded LLM lm_eval dataset with %d samples", len(dataset))
        return {ARTIFACT_TEXT_DECODER: dataset}
