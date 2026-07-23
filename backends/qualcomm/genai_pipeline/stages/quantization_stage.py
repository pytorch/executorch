# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.genai_pipeline.configs.quantization_input_config import (
    QuantizationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.quantization_output_config import (
    QuantizationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.pipeline_stage import PipelineStage
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
    STAGE_QUANTIZATION,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)


class QuantizationStage(PipelineStage):

    def __init__(self, strategy: QuantizationStrategy) -> None:
        self._strategy = strategy

    @property
    def name(self) -> str:
        return STAGE_QUANTIZATION

    def invoke(
        self,
        context: PipelineContext,
        input_config: QuantizationInputConfig,
    ) -> QuantizationOutputConfig:
        return self._strategy.invoke(context, input_config)
