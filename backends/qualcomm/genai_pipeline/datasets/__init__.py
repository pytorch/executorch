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
"""

from executorch.backends.qualcomm.genai_pipeline.datasets.calibration_data_adapter import (
    CalibrationDataAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.default_calibration_data_adapter import (
    DefaultCalibrationDataAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.default_training_data_adapter import (
    DefaultTrainingDataAdapter,
)
from executorch.backends.qualcomm.genai_pipeline.datasets.training_data_adapter import (
    TrainingDataAdapter,
)

__all__ = [
    "CalibrationDataAdapter",
    "DefaultCalibrationDataAdapter",
    "DefaultTrainingDataAdapter",
    "TrainingDataAdapter",
]
