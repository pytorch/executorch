# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_input_config import (
    ModelPreparationInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.model_preparation_output_config import (
    ModelPreparationOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.pipeline_stage import PipelineStage
from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
    STAGE_MODEL_PREPARATION,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.model_preparation_strategy import (
    ModelPreparationStrategy,
)


class ModelPreparationStage(PipelineStage):

    def __init__(self, strategy: ModelPreparationStrategy) -> None:
        self._strategy = strategy

    @property
    def name(self) -> str:
        return STAGE_MODEL_PREPARATION

    def invoke(
        self,
        context: PipelineContext,
        input_config: ModelPreparationInputConfig,
    ) -> ModelPreparationOutputConfig:
        return self._strategy.invoke(context, input_config)
