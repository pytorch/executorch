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
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)


class ExecuTorchQuantizationStrategy(QuantizationStrategy):
    """ExecuTorch-based quantization using QNN quantizer annotator rules.

    Reads backend_type from the input config to select the appropriate annotator (HTP, GPU, or LPAI rules).
    """

    def invoke(
        self,
        context: PipelineContext,
        input_config: QuantizationInputConfig,
    ) -> QuantizationOutputConfig:
        """Quantize the model using ExecuTorch/QNN quantization.

        Args:
            context: The pipeline context.
            input_config: The quantization input configuration.

        Returns:
            QuantizationOutputConfig with quantized model.
        """
        raise NotImplementedError(
            "ExecuTorchQuantizationStrategy.invoke() is a stub. "
            "Implementation will be added in a subsequent PR."
        )
