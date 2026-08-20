# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.genai_pipeline.strategies.inference.executorch_inference_strategy import (
    ExecuTorchInferenceStrategy,
)
from executorch.backends.qualcomm.genai_pipeline.strategies.inference.inference_strategy import (
    InferenceStrategy,
)

__all__ = ["ExecuTorchInferenceStrategy", "InferenceStrategy"]
