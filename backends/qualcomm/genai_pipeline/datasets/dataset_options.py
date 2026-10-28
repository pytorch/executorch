# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DatasetOptions: the typed calibration / training data configuration.

Datasets are a **cross-stage** concern -- the same corpus feeds PTQ calibration
during quantization and result evaluation during inference -- so the knobs that
select and shape that corpus are grouped here at package level rather than nested
inside any one stage's options. Quantization consumes them today; the evaluation
path will consume the same type.

Field names match ``llama.py``'s / the GenAI CLI parser ``dest`` names, so
:meth:`from_namespace` can build a ``DatasetOptions`` straight from a parsed
argument namespace, dropping attributes without a matching field.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, fields
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetOptions:
    """Typed calibration / training / evaluation data configuration.

    Every field selects or shapes the corpus that feeds one part of the flow;
    the grouping below follows which part consumes it. Field names match
    ``llama.py``'s / the GenAI CLI parser ``dest`` names, so
    :meth:`from_namespace` builds a ``DatasetOptions`` straight from a parsed
    namespace, dropping attributes without a matching field.
    """

    # --- Calibration data selection ---
    #
    # ``calib_tasks`` (lm_eval task names) and ``calib_samples`` (message-sample
    # JSON files) are the corpus sources; ``calib_hf_dataset`` adds a HuggingFace
    # chat dataset (e.g. ``HuggingFaceTB/smol-smoltalk``). The ``*_limit`` and
    # ``num_fewshot`` fields shape how much of each is drawn.
    calib_tasks: Optional[List[str]] = None
    calib_samples: Optional[List[str]] = None
    calib_limit: Optional[int] = None
    calib_num_fewshot: Optional[int] = None
    calib_hf_dataset: Optional[str] = None
    calib_hf_limit: int = 1

    # --- Sample shape---
    max_context_len: int = 1024
    batch_size: int = 1
    model_mode: Optional[str] = None

    # --- Quantization mode ---
    qat: bool = False

    # --- Evaluation ---
    eval_tasks: Optional[List[str]] = None
    eval_limit: int = 1
    eval_num_fewshot: Optional[int] = None

    # --- QAT training data ---
    #
    # ``train_val_ratio`` is the fraction of non-calib samples used for training;
    # the remainder becomes validation, and ``1.0`` disables validation.
    train_tasks: Optional[List[str]] = None
    train_limit: int = 1
    train_hf_dataset: Optional[str] = None
    train_hf_limit: int = 1000
    train_val_ratio: float = 1.0

    # --- Random fallback (DefaultCalibrationDataAdapter) ---
    num_samples: Optional[int] = None
    seed: Optional[int] = None

    # --- Non-CLI ---
    #
    # ``llm_config`` is the model's ``LLMModelConfig``, forwarded to adapters
    # that need model-specific tokenization details. It has no CLI argument and
    # is set by the caller after :meth:`from_namespace` (e.g. via
    # ``dataclasses.replace``).
    llm_config: Any = None

    @classmethod
    def field_names(cls) -> frozenset:
        """The set of field names this dataclass defines."""
        return frozenset(f.name for f in fields(cls))

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "DatasetOptions":
        """Build from an ``argparse.Namespace``, ignoring unknown attributes.

        The CLI parser produces a superset of these fields; attributes without a
        matching field are dropped. ``llm_config`` has no CLI argument and is set
        by the caller afterwards (e.g. via ``dataclasses.replace``).

        Args:
            namespace: A parsed namespace, e.g. from the GenAI CLI parser.

        Returns:
            A ``DatasetOptions`` carrying every recognised attribute.
        """
        known = cls.field_names()
        supplied = vars(namespace)

        ignored = sorted(set(supplied) - known)
        if ignored:
            logger.debug("Ignoring unrecognised arguments: %s", ignored)

        return cls(**{k: v for k, v in supplied.items() if k in known})
