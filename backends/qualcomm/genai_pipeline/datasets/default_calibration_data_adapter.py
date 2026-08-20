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
DEFAULT_SEED = 42


class DefaultCalibrationDataAdapter:
    """Default calibration data provider using random token sequences.

    Suitable for bring-up. For production quantization accuracy, pass a real
    corpus via ``extra_options["dataset"]`` or inject a corpus-backed adapter
    (HuggingFace prompts, ``lm_eval`` tasks, JSON prompt files, ...).
    """

    def generate_calibration_data(
        self,
        tokenizer: Any,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        seq_length: int = DEFAULT_SEQ_LENGTH,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Any]:
        """Generate calibration data as random token sequences.

        Args:
            tokenizer: The tokenizer to use. Only ``vocab_size`` is consulted.
            num_samples: Number of calibration samples to generate.
            seq_length: Sequence length for each sample.
            extra_options: Additional options. Supported keys:
                - ``dataset``: Caller-supplied calibration data to use instead
                  of random data. Accepts any ``Iterable[Tuple[Tensor, ...]]``,
                  including a ``DataLoader`` with a custom ``collate_fn``, which
                  allows batched/streamed calibration without materializing the
                  whole dataset.
                - ``seed``: Random seed for reproducibility (default: 42).

        Returns:
            A list of ``(input_ids, attention_mask)`` tuples of shape
            ``(1, seq_length)``.

        Raises:
            ValueError: If the tokenizer has no usable ``vocab_size`` and no
                ``dataset`` was supplied.
        """
        import torch

        extra_options = extra_options or {}

        # A caller-supplied dataset takes precedence over random generation.
        if "dataset" in extra_options:
            logger.info("Using caller-supplied dataset for calibration")
            return extra_options["dataset"]

        seed = extra_options.get("seed", DEFAULT_SEED)
        torch.manual_seed(seed)

        logger.info(
            "Generating %d random calibration samples (seq_length=%d, seed=%d)",
            num_samples,
            seq_length,
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
        for _ in range(num_samples):
            input_ids = torch.randint(0, vocab_size, (1, seq_length))
            attention_mask = torch.ones(1, seq_length, dtype=torch.long)
            calibration_data.append((input_ids, attention_mask))

        return calibration_data
