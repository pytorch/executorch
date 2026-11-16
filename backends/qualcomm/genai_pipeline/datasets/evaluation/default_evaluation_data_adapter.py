# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_NUM_SAMPLES = 128
DEFAULT_SEQ_LENGTH = 1024


class DefaultEvaluationDataAdapter:
    """Post-quantization evaluation data provider.

    Evaluation data is not synthesized. Uses injected dataset loaders when
    present; otherwise callers must supply data via ``extra_options["eval_data"]``.

    The returned object is always ``{component: iterable}``.

    Args:
        dataset_loaders: Purpose-agnostic loaders, each exposing
            ``load_dataset`` and returning a component-keyed dataset dict.
        collector: Provides per-component collators for wrapping datasets in
            DataLoaders.
    """

    def __init__(
        self,
        dataset_loaders: Optional[List[Any]] = None,
        collector: Optional[Any] = None,
    ) -> None:
        self._dataset_loaders = dataset_loaders or []
        self._collector = collector

    def generate_eval_data(
        self,
        tokenizer: Any,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        seq_length: int = DEFAULT_SEQ_LENGTH,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Iterable[Any]]:
        """Return component-keyed post-quantization evaluation data."""
        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )
        from torch.utils.data import ConcatDataset, DataLoader

        extra_options = extra_options or {}

        eval_data = extra_options.get("eval_data")
        if eval_data is not None:
            logger.info("Using caller-supplied evaluation data")
            return (
                eval_data
                if isinstance(eval_data, dict)
                else {ARTIFACT_TEXT_DECODER: eval_data}
            )

        if not self._dataset_loaders:
            raise ValueError(
                "No evaluation data supplied. DefaultEvaluationDataAdapter does not "
                "synthesize evaluation data; pass it via "
                "extra_options['eval_data'] or inject corpus sources."
            )

        if self._collector is None:
            raise ValueError("collector is required when dataset_loaders are provided")

        max_context_len = extra_options.get("max_context_len", seq_length)
        example_inputs = extra_options.get("example_inputs")
        batch_size = extra_options.get("batch_size", 1)

        grouped: Dict[str, List[Any]] = {}
        for loader in self._dataset_loaders:
            result = loader.load_dataset(
                tokenizer=tokenizer,
                max_context_len=max_context_len,
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
            return {}

        collators = self._collector.create_collators(example_inputs, max_context_len)

        dataloaders: Dict[str, Iterable[Any]] = {}
        for component, component_dataset in merged.items():
            component_batch_size = (
                batch_size if component == ARTIFACT_TEXT_DECODER else 1
            )
            drop_last = component == ARTIFACT_TEXT_DECODER and component_batch_size > 1
            dataloaders[component] = DataLoader(
                component_dataset,
                batch_size=component_batch_size,
                shuffle=False,
                drop_last=drop_last,
                collate_fn=collators[component],
            )
        return dataloaders
