# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.genai_pipeline.configs.inference_input_config import (
    InferenceInputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.pipeline_context import PipelineContext
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)


class ExecuTorchInferenceStrategy(InferenceStrategy):
    """ExecuTorch-based inference using QNN runtime on device."""

    def invoke(
        self,
        context: PipelineContext,
        input_config: InferenceInputConfig,
    ) -> InferenceOutputConfig:
        raise NotImplementedError(
            "ExecuTorchInferenceStrategy.invoke() is a stub. "
            "Implementation will be added in a subsequent PR."
        )
