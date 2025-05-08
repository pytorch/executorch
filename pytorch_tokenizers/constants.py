# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT

CL100K_PAT_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

LLAMA_BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",  # end of message
    "<|eot_id|>",  # end of turn
    "<|python_tag|>",
    "<|image|>",
]

LLAMA_NUM_RESERVED_SPECIAL_TOKENS = 256
LLAMA_RESERVED_SPECIAL_TOKENS = [
    f"<|reserved_special_token_{2 + i}|>"
    for i in range(LLAMA_NUM_RESERVED_SPECIAL_TOKENS - len(LLAMA_BASIC_SPECIAL_TOKENS))
]

LLAMA_SPECIAL_TOKENS = LLAMA_BASIC_SPECIAL_TOKENS + LLAMA_RESERVED_SPECIAL_TOKENS
