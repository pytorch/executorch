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


class MLLMCalibrationDataAdapter:
    """Corpus-backed calibration provider for multimodal models (VLM/ALM).

    Composes the injected dataset loaders into a single dataset per component
    (ARTIFACT_TEXT_DECODER plus ARTIFACT_VISION_ENCODER / ARTIFACT_AUDIO_ENCODER) and wraps each in a
    DataLoader using the collator the collector supplies. The text decoder is
    batched; encoder components stay at batch size 1.

    The returned object is always ``{component: DataLoader}``.

    Args:
        dataset_loaders: Purpose-agnostic loaders, each exposing
            ``load_dataset`` and returning a component-keyed dataset dict.
        collector: Provides per-component collators for wrapping datasets in
            DataLoaders.
        max_context_len: Sequence length for encoding and padding. The single
            source of truth: it is pushed into ``extra_options`` before the
            loaders run, so their encoding length and the collator's padding
            length cannot diverge.
        batch_size: Batch size for the ``ARTIFACT_TEXT_DECODER`` DataLoader; encoder
            components stay at batch size 1.
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
        """Return a component-keyed DataLoader map built from the loaders.

        Args:
            tokenizer: Tokenizer passed to each configured dataset loader.
            example_inputs: Calibration-graph signatures required to construct
                the component collators.
            extra_options: Optional settings forwarded to each dataset loader.

        Returns:
            A map from artifact component keys to calibration ``DataLoader``s.

        Note:
            ``max_context_len`` and ``batch_size`` are fixed at construction.
            The context length is passed to both dataset loaders and the
            collators so their sequence shapes remain aligned.
        """
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )
        from torch.utils.data import ConcatDataset, DataLoader

        extra_options = extra_options or {}

        if "dataset" in extra_options:
            logger.info("Using caller-supplied dataset for calibration")
            data = extra_options["dataset"]
            return data if isinstance(data, dict) else {ARTIFACT_TEXT_DECODER: data}

        if not self._dataset_loaders:
            raise ValueError(
                "MLLMCalibrationDataAdapter requires at least one dataset loader; "
                "use DefaultCalibrationDataAdapter for the random fallback."
            )
        if self._collector is None:
            raise ValueError("collector is required when dataset_loaders are provided")

        grouped: Dict[str, List[Any]] = {}
        for loader in self._dataset_loaders:
            result = loader.load_dataset(
                tokenizer=tokenizer,
                max_context_len=self._max_context_len,
                extra_options=extra_options,
            )
            for component, dataset in result.items():
                if dataset is not None:
                    grouped.setdefault(component, []).append(dataset)

        merged = {
            component: (datasets[0] if len(datasets) == 1 else ConcatDataset(datasets))
            for component, datasets in grouped.items()
        }
        if not merged:
            raise ValueError(
                "No calibration dataset produced: the configured loaders "
                "returned no data for any component."
            )

        collators = self._collector.create_collators(
            example_inputs, self._max_context_len
        )

        dataloaders: Dict[str, Iterable[Any]] = {}
        for component, component_dataset in merged.items():
            dataloaders[component] = DataLoader(
                component_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                drop_last=self._batch_size > 1,
                collate_fn=collators[component],
            )
        return dataloaders
