# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT

import json
import os
from typing import List, Optional

from tokenizers import Tokenizer


class HuggingFaceTokenizer:
    """
    Tokenizing and encoding/decoding text using the Hugging face tokenizer.
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initializes the Tokenizer with a tokenizer.json from HuggingFace.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        self.model = tokenizer = Tokenizer.from_file(model_path)

        self.n_words: int = tokenizer.get_vocab_size()
        if config_path:
            with open(config_path) as f:
                tokenizer_config = json.load(f)
                self.bos_id = (
                    self.model.token_to_id(tokenizer_config["bos_token"])
                    if tokenizer_config["bos_token"]
                    else None
                )
                self.eos_id = self.model.token_to_id(tokenizer_config["eos_token"])
        else:  # Fallback guess.
            self.bos_id = self.model.token_to_id("<|begin_of_text|>")
            self.eos_id = self.model.token_to_id("<|endoftext|>")

        self.stop_tokens = [
            self.eos_id,
        ]

    def encode(self, s: str, *, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        return self.model.encode(s).ids

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t)

    def decode_token(self, t: int) -> str:
        return self.model.decode([t])
