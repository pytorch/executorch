# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Optional

from executorch.examples.models.llama.tokenizer.tiktoken import Tokenizer as Tiktoken
from executorch.extension.llm.tokenizer.hf_tokenizer import HFTokenizer
from executorch.extension.llm.tokenizer.tokenizer import (
    Tokenizer as SentencePieceTokenizer,
)


def get_tokenizer(tokenizer_path: str, tokenizer_config_path: Optional[str] = None):
    if tokenizer_path.endswith(".json"):
        from tokenizers import Tokenizer

        # Load the tokenizer from the tokenizer.json file
        tokenizer = Tokenizer.from_file(tokenizer_path)

        # export_llama expects n_words attribute.
        tokenizer.n_words = tokenizer.get_vocab_size()
        # Keep in line with internal tokenizer apis.
        tokenizer.decode_token = lambda token: tokenizer.decode([token])

        if tokenizer_config_path:
            with open(tokenizer_config_path) as f:
                tokenizer_config = json.load(f)
                tokenizer.eos_id = tokenizer.token_to_id(tokenizer_config["eos_token"])
    else:
        try:
            tokenizer = SentencePieceTokenizer(model_path=str(tokenizer_path))
        except Exception:
            print("Using Tiktokenizer")
            tokenizer = Tiktoken(model_path=str(tokenizer_path))
    return tokenizer
