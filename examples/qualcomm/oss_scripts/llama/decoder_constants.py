# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

EVAL_MODE = {
    "kv": 0,
    "hybrid": 1,
    "lookahead": 2,
}

# The dict's value is mainly for runner to decide what special tokens are required to wrap the prompt.
DECODER_MODEL_VERSION = {
    "stories260k": "llama2",
    "stories110m": "llama2",
    "gemma3-1b": "gemma3",
    "phi_4_mini": "phi_4_mini",
    "llama3_2-1b_instruct": "llama3",
    "llama3_2-3b_instruct": "llama3",
    "qwen2_5-0_5b": "qwen2_5",
    "qwen2_5-1_5b": "qwen2_5",
    "qwen3-0_6b": "qwen3",
    "qwen3-1_7b": "qwen3",
    "smollm2_135m": "smollm2_135m",
    "smollm3-3b": "smollm3",
}
