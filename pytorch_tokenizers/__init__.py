# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT


from typing import Optional

from .hf_tokenizer import HuggingFaceTokenizer
from .llama2c import Llama2cTokenizer
from .tiktoken import TiktokenTokenizer

__all__ = ["TiktokenTokenizer", "Llama2cTokenizer", "HuggingFaceTokenizer"]


def get_tokenizer(tokenizer_path: str, tokenizer_config_path: Optional[str] = None):
    if tokenizer_path.endswith(".json"):
        tokenizer = HuggingFaceTokenizer(tokenizer_path, tokenizer_config_path)
    else:
        try:
            tokenizer = Llama2cTokenizer(model_path=str(tokenizer_path))
        except Exception:
            print("Using Tiktokenizer")
            tokenizer = TiktokenTokenizer(model_path=str(tokenizer_path))
    return tokenizer
