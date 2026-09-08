# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable, Tuple


@runtime_checkable
class TrainingDataAdapter(Protocol):
    """Protocol for quantization-aware training (QAT) dataset construction.

    Separate from :class:`CalibrationDataAdapter` because the contracts differ:
    calibration yields model-argument tuples that get splatted as
    ``model(*sample)``, whereas training yields ``(features, labels)`` pairs --
    labels are required to compute a loss and have no analogue in PTQ.
    """

    def generate_training_data(
        self,
        tokenizer: Any,
        num_samples: int = 128,
        seq_length: int = 1024,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Tuple[Any, Any]]:
        """Build the training dataset used for quantization-aware training.

        Args:
            tokenizer: The tokenizer used to encode text samples.
            num_samples: Number of training samples to produce.
            seq_length: Sequence length for each sample.
            extra_options: Implementation-specific options. Implementations may
                support a ``training_data`` key accepting caller-supplied data
                directly, including a ``DataLoader``.

        Returns:
            An iterable of ``(features, labels)`` pairs. ``features`` is a tuple
            of model inputs; ``labels`` holds the ground truth. Mirrors the
            ``qat_training_data`` contract in ``build_executorch_binary``.
        """
        ...
