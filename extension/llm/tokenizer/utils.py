# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.models.llama.tokenizer.tiktoken import Tokenizer as Tiktoken
from executorch.extension.llm.tokenizer.tokenizer import (
    Tokenizer as SentencePieceTokenizer,
)
from executorch.extension.llm.tokenizer.hf_tokenizer import HFTokenizer


def get_tokenizer(tokenizer_path):
    if tokenizer_path.endswith(".json"):
        # print("Using Hugging Face tokenizer")
        # tokenizer = HFTokenizer()
        # tokenizer.load(tokenizer_path)

        from tokenizers import Tokenizer

        # Load the tokenizer from the tokenizer.json file
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # from tokenizers import SentencePieceBPETokenizer

        # tokenizer = SentencePieceBPETokenizer(tokenizer_path)
        tokenizer.n_words = tokenizer.get_vocab_size()
        breakpoint()
    else:
        try:
            tokenizer = SentencePieceTokenizer(model_path=str(tokenizer_path))
        except Exception:
            print("Using Tiktokenizer")
            tokenizer = Tiktoken(model_path=str(tokenizer_path))
    return tokenizer
