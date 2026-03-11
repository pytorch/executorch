# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from transformers.models.phi3.configuration_phi3 import Phi3Config


def get_phi3_test_config() -> Phi3Config:
    config = Phi3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        original_max_position_embeddings=32,
        use_cache=False,
        tie_word_embeddings=False,
    )
    # Force eager attention path to keep the module exportable in tests.
    config._attn_implementation = "eager"
    return config
