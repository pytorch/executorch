# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext


class ModelPreparationStrategy(ABC):
    @abstractmethod
    def invoke(
        self,
        context: PipelineContext,
        input_config: ModelPreparationInputConfig,
    ) -> ModelPreparationOutputConfig:
        """Prepare the model, tokenizer, and calibration data.

        Args:
            context: The pipeline context with global settings.
            input_config: The model preparation input configuration.

        Returns:
            ModelPreparationOutputConfig with model, tokenizer, and calibration data.
        """
        ...
