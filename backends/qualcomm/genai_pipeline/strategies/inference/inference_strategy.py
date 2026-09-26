# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext


class InferenceStrategy(ABC):
    @abstractmethod
    def invoke(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
    ) -> InferenceOutputConfig:
        """Run inference using the compiled model artifacts.

        Args:
            context: The pipeline context with global settings.
            input_config: The inference input configuration.

        Returns:
            InferenceOutputConfig with inference results and metrics.
        """
        ...
