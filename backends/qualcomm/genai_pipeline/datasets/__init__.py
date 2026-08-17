# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset providers for the GenAI pipeline.

Datasets are a **cross-stage** concern rather than a model-preparation detail:
the same corpus feeds PTQ calibration during quantization and on-device result
evaluation during inference (including pre-built ``.pte`` flows, where no model
preparation runs at all). They therefore live here rather than under
``strategies/model_preparation/``.

Dataset loaders are organized by modality under ``loaders/`` (``loaders/llm/``,
``loaders/mllm/``) and are purpose-agnostic: each just loads a component-keyed
dataset dict. The purpose lives in the per-purpose adapter packages
(``calibration/``, ``training/``, ``evaluation/``), which compose a set of
loaders with a component-aware collector.
"""

from executorch.backends.qualcomm.genai_pipeline.datasets.dataset_lookup import (
    get_calibration_dataset_adapter,
    get_collector,
    get_dataset_adapter,
    get_eval_dataset_adapter,
    get_training_dataset_adapter,
)

__all__ = [
    "get_dataset_adapter",
    "get_collector",
    "get_calibration_dataset_adapter",
    "get_training_dataset_adapter",
    "get_eval_dataset_adapter",
]
