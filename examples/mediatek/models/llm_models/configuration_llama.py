# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import nullcontext

from models.llm_models.configuration_base import BaseConfig


# flake8: noqa: C901


class LlamaConfig(BaseConfig):
    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        intermediate_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        max_position_embeddings=None,
        norm="RMSNorm",
        position_embedding="rope",
        norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=0,
        use_stable_embedding=False,
        tie_word_embeddings=False,
        combine_qkv=False,
        response_handler=None,
        **kwargs,
    ):
        super().__init__()

        self.model_type = "llama"
        self.vocab_size = vocab_size
        if self.vocab_size is None:
            raise KeyError("vocab_size is required but missing from config.json")
        self.hidden_size = hidden_size
        if self.hidden_size is None:
            raise KeyError("hidden_size is required but missing from config.json")
        self.intermediate_size = intermediate_size
        if self.intermediate_size is None:
            raise KeyError("intermediate_size is required but missing from config.json")
        self.num_hidden_layers = num_hidden_layers
        if self.num_hidden_layers is None:
            raise KeyError("num_hidden_layers is required but missing from config.json")
        self.num_attention_heads = num_attention_heads
        if self.num_attention_heads is None:
            raise KeyError(
                "num_attention_heads is required but missing from config.json"
            )
        self.num_key_value_heads = kwargs.pop(
            "num_key_value_heads", self.num_attention_heads
        )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise RuntimeError(
                f"num_attention_heads ({self.num_attention_heads}) must be exactly "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        if norm not in ["RMSNorm", "LayerNorm"]:
            raise ValueError("norm must be one of: RMSNorm (default) or LayerNorm")
        self.norm = norm
        self.norm_eps = kwargs.pop("rms_norm_eps", norm_eps)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        if position_embedding not in ["rope", "alibi"]:
            raise ValueError("Positional embedding must be one of: rope, alibi")
        self.position_embedding = position_embedding
        self.ntk_scaling_factor = kwargs.pop("ntk_scaling_factor", 1.0)
        if self.ntk_scaling_factor != 1.0 and self.position_embedding != "rope":
            raise KeyError("ntk_scaling_factor is strictly for position_embedding=rope")
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings is None and self.position_embedding == "rope":
            raise KeyError(
                "max_position_embeddings is required for position_embedding=rope but missing from config.json"
            )

        self.use_stable_embedding = use_stable_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.combine_qkv = combine_qkv

        self.tokenizer = kwargs.pop("tokenizer", self.tokenizer)

        if response_handler is None:
            response_handler = nullcontext()
        if kwargs.pop("verbose", True):
            self.print_config(response_handler)

    def print_config(self, response_handler):
        with response_handler:
            print(f"{self.model_type} config:")
            print(f"Hidden size:          {self.hidden_size}")
            print(f"Intermediate size:    {self.intermediate_size}")
            print(f"Num layers:           {self.num_hidden_layers}")
            print(f"Num attention heads:  {self.num_attention_heads}")
            print(f"Num KV heads:         {self.num_key_value_heads}")
            print(f"Positional embedding: {self.position_embedding}")
            if self.position_embedding == "rope":
                print(f"Max pos emb:          {self.max_position_embeddings}")
                if self.ntk_scaling_factor != 1.0:
                    print(f"NTK scaling factor:   {self.ntk_scaling_factor}")
            print(f"Norm type:            {self.norm}")
            print(f"Norm epsilon:         {self.norm_eps}")
            print(f"BOS token id:         {self.bos_token_id}")
            print(f"EOS token id:         {self.eos_token_id}")
            print(f"PAD token id:         {self.pad_token_id}")
            print(f"UNK token id:         {self.unk_token_id}")
            print(f"Vocab size:           {self.vocab_size}")
            print(f"Use stable embedding: {self.use_stable_embedding}")
            print(f"Tie word embeddings:  {self.tie_word_embeddings}")
            print(f"Combine QKV:          {self.combine_qkv}")
            if self.tokenizer != "default":
                print(f"Tokenizer:            {self.tokenizer}")
            print()
