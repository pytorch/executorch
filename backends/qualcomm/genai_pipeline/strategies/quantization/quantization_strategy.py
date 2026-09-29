# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext


class QuantizationStrategy(ABC):
    @abstractmethod
    def invoke(
        self,
        context: PipelineContext,
        input_config: QuantizationInputConfig,
    ) -> QuantizationOutputConfig:
        """Quantize the model.

        Args:
            context: The pipeline context with global settings.
            input_config: The quantization input configuration.

        Returns:
            QuantizationOutputConfig with quantized model.
        """
        ...
