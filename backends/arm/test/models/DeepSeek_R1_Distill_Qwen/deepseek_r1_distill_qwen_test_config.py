# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


def get_deepseek_r1_distill_qwen_1_5b_checkpoint_config() -> Qwen2Config:
    return Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_act="silu",
        hidden_size=1536,
        initializer_range=0.02,
        intermediate_size=8960,
        max_position_embeddings=131072,
        max_window_layers=21,
        num_attention_heads=12,
        num_hidden_layers=28,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,  # type: ignore[arg-type]
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        sliding_window=4096,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        transformers_version="4.44.0",
        use_cache=True,
        use_mrope=False,
        use_sliding_window=False,
        vocab_size=151936,
    )
