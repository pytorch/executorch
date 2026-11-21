# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flag values the llama-family runners take, mirroring the legacy flow's."""

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_VISION_ENCODER,
)

# ``--eval_mode`` value of each ``--model-mode``.
EVAL_MODE = {
    "kv": 0,
    "hybrid": 1,
    "lookahead": 2,
}

# ``--decoder_model_version`` of each decoder model, which decides the special
# tokens the runner wraps the prompt with.
DECODER_MODEL_VERSION = {
    "stories260k": "llama2",
    "stories110m": "llama2",
    "llama3_2-1b_instruct": "llama3",
    "llama3_2-3b_instruct": "llama3",
    "codegen2_1b": "codegen",
    "gemma-2b": "gemma",
    "gemma2-2b": "gemma2",
    "gemma3-1b": "gemma3",
    "granite_3_3-2b_instruct": "granite",
    "granite_speech_3_3-2b": "granite_speech",
    "phi_4_mini": "phi_4_mini",
    "qwen2_5-0_5b": "qwen2_5",
    "qwen2_5-1_5b": "qwen2_5",
    "qwen3-0_6b": "qwen3",
    "qwen3-1_7b": "qwen3",
    "smollm2_135m": "smollm2_135m",
    "smollm3-3b": "smollm3",
    "glm-1_5b": "glm",
    "smolvlm_500m_instruct": "smolvlm",
    "internvl3_1b": "internvl3",
    "gemma4-e2b": "gemma4",
}

# The multimodal runner flag that takes each modality encoder's inputs.
MODALITY_INPUT_FLAG_MAP = {
    ARTIFACT_AUDIO_ENCODER: "audio_path",
    ARTIFACT_VISION_ENCODER: "image_path",
}
