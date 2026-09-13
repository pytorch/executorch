# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_NUM_SAMPLES = 128
DEFAULT_SEQ_LENGTH = 1024


class DefaultTrainingDataAdapter:
    """Pass-through QAT training data provider.

    Training data is **not** synthesized: random tokens carry no learning
    signal, so there is no meaningful default corpus. Callers enabling QAT must
    supply their data via ``extra_options["training_data"]``, or inject a
    corpus-backed :class:`TrainingDataAdapter`.
    """

    def generate_training_data(
        self,
        tokenizer: Any,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        seq_length: int = DEFAULT_SEQ_LENGTH,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Tuple[Any, Any]]:
        """Return caller-supplied QAT training data.

        Args:
            tokenizer: The tokenizer (unused by this implementation).
            num_samples: Ignored; retained for protocol conformance.
            seq_length: Ignored; retained for protocol conformance.
            extra_options: Additional options. Supported keys:
                - ``training_data``: ``(features, labels)`` pairs, or any
                  ``Iterable`` yielding them (including a ``DataLoader``).

        Returns:
            The caller-supplied ``(features, labels)`` pairs.

        Raises:
            ValueError: If no ``training_data`` was supplied. QAT cannot proceed
                without labels, so failing loudly is preferable to silently
                degrading to PTQ.
        """
        extra_options = extra_options or {}

        training_data = extra_options.get("training_data")
        if training_data is None:
            raise ValueError(
                "No training data supplied. DefaultTrainingDataAdapter does not "
                "synthesize labelled data; pass it via "
                "extra_options['training_data'] or inject a corpus-backed "
                "TrainingDataAdapter. For PTQ, skip QAT entirely and use "
                "CalibrationDataAdapter instead."
            )

        logger.info("Using caller-supplied training data for QAT")
        return training_data
