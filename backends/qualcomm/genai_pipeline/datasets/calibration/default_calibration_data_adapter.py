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
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 42


class DefaultCalibrationDataAdapter:
    """Pipeline sanity-check calibration provider.

    Returns caller-supplied data when present; otherwise random token
    sequences. This adapter deliberately does **not** wrap datasets in
    DataLoaders or run them through a collator -- it exists to confirm the
    quantization pipeline is wired up, not to produce corpus-backed calibration.
    For production quantization accuracy, use the modality-specific
    ``LLMCalibrationDataAdapter`` / ``MLLMCalibrationDataAdapter`` (selected by
    ``get_calibration_dataset_adapter`` when dataset sources are configured) or
    pass ready-made data via ``extra_options["dataset"]``.

    The returned object is always ``{component: iterable}``.

    Args:
        num_samples: How many random samples to generate.
        batch_size: Leading batch dimension of each random sample.
        seed: Manual seed for reproducible generation.
    """

    def __init__(
        self,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._num_samples = num_samples
        self._batch_size = batch_size
        self._seed = seed

    def generate_calibration_data(
        self,
        tokenizer: Any,
        example_inputs: Optional[Dict[str, Any]] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Iterable[Any]]:
        """Return caller-supplied data or random fallback data.

        Args:
            tokenizer: Tokenizer whose vocabulary size bounds random token IDs.
            example_inputs: Ignored because this adapter does not use collators.
            extra_options: Optional calibration settings. ``dataset`` supplies
                ready-made calibration data; ``max_context_len`` and ``seed``
                configure random fallback data.

        Returns:
            A map containing caller-supplied data or random decoder inputs.

        Note:
            Random samples are ``(input_ids, attention_mask)`` tuples keyed by
            ``ARTIFACT_TEXT_DECODER`` and can be consumed directly without a
            collator. ``DEFAULT_SEQ_LENGTH`` is used when no sequence length is
            supplied.
        """
        import torch

        from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
            ARTIFACT_TEXT_DECODER,
        )

        extra_options = extra_options or {}

        if "dataset" in extra_options:
            logger.info("Using caller-supplied dataset for calibration")
            data = extra_options["dataset"]
            return data if isinstance(data, dict) else {ARTIFACT_TEXT_DECODER: data}

        seq_length = extra_options.get("max_context_len", DEFAULT_SEQ_LENGTH)
        seed = extra_options.get("seed", self._seed)
        torch.manual_seed(seed)

        logger.info(
            "Generating %d random calibration samples (seq_length=%d, batch_size=%d, seed=%d)",
            self._num_samples,
            seq_length,
            self._batch_size,
            seed,
        )

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None or vocab_size <= 0:
            raise ValueError(
                "Tokenizer does not have a valid vocab_size attribute. "
                "Cannot generate random calibration data. Supply a dataset "
                "via extra_options['dataset'] instead."
            )

        calibration_data: List[Any] = []
        for _ in range(self._num_samples):
            input_ids = torch.randint(0, vocab_size, (self._batch_size, seq_length))
            attention_mask = torch.ones(self._batch_size, seq_length, dtype=torch.long)
            calibration_data.append((input_ids, attention_mask))

        return {ARTIFACT_TEXT_DECODER: calibration_data}
