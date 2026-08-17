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
from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.model_preparation_strategy import (
    ModelPreparationStrategy,
)


class ExecuTorchModelPreparationStrategy(ModelPreparationStrategy):
    """ExecuTorch-based model preparation using HuggingFace transformers.

    Loads model weights, tokenizer, and generates calibration data
    for downstream quantization and compilation stages.
    """

    def invoke(
        self,
        context: PipelineContext,
        input_config: ModelPreparationInputConfig,
    ) -> ModelPreparationOutputConfig:
        """Prepare the model, tokenizer, and calibration data.

        Args:
            context: The pipeline context.
            input_config: The model preparation input configuration.

        Returns:
            ModelPreparationOutputConfig with model, tokenizer, and calibration data.
        """
        raise NotImplementedError(
            "ExecuTorchModelPreparationStrategy.invoke() is a stub. "
            "Implementation will be added in a subsequent PR."
        )
