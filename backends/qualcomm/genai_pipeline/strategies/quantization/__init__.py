# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.executorch_quantization_strategy import (
    ExecuTorchQuantizationStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.quantization_strategy import (
    QuantizationStrategy,
)

__all__ = ["ExecuTorchQuantizationStrategy", "QuantizationStrategy"]
