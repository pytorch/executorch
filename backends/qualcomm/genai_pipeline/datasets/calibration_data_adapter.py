# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class CalibrationDataAdapter(Protocol):
    """Protocol for calibration dataset construction.

    Kept separate from :class:`ModelLoaderAdapter` because data sources
    (HuggingFace prompts, ``lm_eval`` tasks, JSON prompt files, multimodal file
    inputs, synthetic tokens) vary independently of how the model is loaded and
    mostly depend only on the tokenizer. Swapping the source is therefore a
    matter of injecting a different adapter.

    The same calibration corpus is also consumed by the inference stage for
    on-device result evaluation -- including pre-built ``.pte`` flows where model
    preparation never runs -- which is why this lives in ``datasets/`` rather
    than under a single stage.

    For quantization-aware training, see :class:`TrainingDataAdapter`: training
    data yields ``(features, labels)`` pairs rather than model-argument tuples,
    so it is a distinct contract.
    """

    def generate_calibration_data(
        self,
        tokenizer: Any,
        num_samples: int = 128,
        seq_length: int = 1024,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Any]:
        """Build the calibration dataset used for post-training quantization.

        Args:
            tokenizer: The tokenizer used to encode text samples.
            num_samples: Number of calibration samples to produce.
            seq_length: Sequence length for each sample.
            extra_options: Implementation-specific options. Implementations may
                support a ``dataset`` key accepting any
                ``Iterable[Tuple[Tensor, ...]]`` (including a ``DataLoader``)
                to use caller-supplied data directly.

        Returns:
            An iterable of calibration input tuples, each suitable for
            ``model(*sample)``.
        """
        ...
