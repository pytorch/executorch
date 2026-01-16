# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# filenames for vision model
VISION_ENCODER_INPUT_FILENAME = "vision_encoder_input"


# Component identifiers
AUDIO_ENCODER = "audio_encoder"
VISION_ENCODER = "vision_encoder"
TEXT_ENCODER = "text_encoder"
TEXT_EMBEDDING = "text_embedding"
TEXT_DECODER = "text_decoder"

# Text embedding graph names
TEXT_EMBEDDING_GRAPH_NAMES = [
    "tok_embedding_kv_forward",
    "tok_embedding_prefill_forward",
]
# Decoder graph names
DECODER_GRAPH_NAMES = ["kv_forward", "prefill_forward"]


# evaluation mode
EVAL_MODE = {
    "kv": 0,
    "hybrid": 1,
    "lookahead": 2,
}
# The dict's value is mainly for runner to decide what special tokens are required to wrap the prompt.
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
}
