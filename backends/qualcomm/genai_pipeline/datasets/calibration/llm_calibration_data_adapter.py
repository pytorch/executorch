# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1


class LLMCalibrationDataAdapter:
    """Corpus-backed calibration provider for text-only LLMs.

    Composes the injected dataset loaders into a single ``ARTIFACT_TEXT_DECODER`` dataset
    and wraps it in a DataLoader using the collator the collector supplies.
    Unlike ``DefaultCalibrationDataAdapter``, this is the production path:
    real token sequences, padded and masked by ``LLMCalibCollator``.

    An LLM has a single ``ARTIFACT_TEXT_DECODER`` component; the multi-component case
    (encoders) belongs to ``MLLMCalibrationDataAdapter``.

    The returned object is always ``{ARTIFACT_TEXT_DECODER: DataLoader}``.

    Args:
        dataset_loaders: Purpose-agnostic loaders, each exposing
            ``load_dataset`` and returning a ``{ARTIFACT_TEXT_DECODER: dataset}`` dict.
        collector: Provides the ``ARTIFACT_TEXT_DECODER`` collator for wrapping the
            dataset in a DataLoader.
        max_context_len: Sequence length for encoding and padding. The single
            source of truth: it is pushed into ``extra_options`` before the
            loaders run, so their encoding length and the collator's padding
            length cannot diverge.
        batch_size: Batch size for the ``ARTIFACT_TEXT_DECODER`` DataLoader.
    """

    def __init__(
        self,
        dataset_loaders: List[Any],
        collector: Any,
        max_context_len: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._dataset_loaders = dataset_loaders
        self._collector = collector
        self._max_context_len = max_context_len
        self._batch_size = batch_size

    def generate_calibration_data(
        self,
        tokenizer: Any,
        example_inputs: Optional[Dict[str, Any]] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Iterable[Any]]:
        """Return ``{ARTIFACT_TEXT_DECODER: DataLoader}`` built from the loaders.

        Args:
            tokenizer: Tokenizer passed to each configured dataset loader.
            example_inputs: Calibration-graph signatures required to construct
                the decoder collator.
            extra_options: Optional settings forwarded to each dataset loader.

        Returns:
            A map containing a decoder calibration ``DataLoader``.

        Note:
            ``max_context_len`` and ``batch_size`` are fixed at construction.
            The context length is passed to both dataset loaders and the
            collator so their sequence shapes remain aligned.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )
        from torch.utils.data import ConcatDataset, DataLoader

        extra_options = extra_options or {}

        if not self._dataset_loaders:
            raise ValueError(
                "LLMCalibrationDataAdapter requires at least one dataset loader; "
                "use DefaultCalibrationDataAdapter for the random fallback."
            )
        if self._collector is None:
            raise ValueError("collector is required when dataset_loaders are provided")

        datasets: List[Any] = []
        for loader in self._dataset_loaders:
            result = loader.load_dataset(
                tokenizer=tokenizer,
                max_context_len=self._max_context_len,
                extra_options=extra_options,
            )
            dataset = result.get(ARTIFACT_TEXT_DECODER)
            if dataset is not None:
                datasets.append(dataset)

        if not datasets:
            raise ValueError(
                "No calibration dataset produced: the configured loaders "
                "returned no ARTIFACT_TEXT_DECODER data."
            )

        merged = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        collators = self._collector.create_collators(
            example_inputs, self._max_context_len
        )

        drop_last = self._batch_size > 1
        return {
            ARTIFACT_TEXT_DECODER: DataLoader(
                merged,
                batch_size=self._batch_size,
                shuffle=False,
                drop_last=drop_last,
                collate_fn=collators[ARTIFACT_TEXT_DECODER],
            )
        }
