# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GenAI Pipeline model registry.

Re-exports LLM model configurations from the existing llama module, providing a
stable import path within the genai_pipeline namespace.

Which transforms each model uses is owned by
``model_lookup.get_source_transform``, not declared here.

Usage:
    from executorch.backends.qualcomm.genai_pipeline.models import (
        LLMModelConfig,
        SUPPORTED_LLM_MODELS,
    )
"""

from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    LLMModelConfig,
    register_llm_model,
    SUPPORTED_LLM_MODELS,
)

__all__ = [
    "LLMModelConfig",
    "LLM_VARIANT_ARCHS",
    "register_llm_model",
    "SUPPORTED_LLM_MODELS",
]
