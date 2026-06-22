# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


def get_qwen3_vl_2b_instruct_checkpoint_config() -> Qwen3VLConfig:
    text_config = Qwen3VLTextConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,  # type: ignore[call-arg]
        dtype="bfloat16",
        eos_token_id=151645,  # type: ignore[call-arg]
        head_dim=128,
        hidden_act="silu",
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=6144,
        max_position_embeddings=262144,
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        rms_norm_eps=1e-6,
        rope_parameters={
            "mrope_interleaved": True,  # type: ignore[dict-item]
            "mrope_section": [24, 20, 20],  # type: ignore[dict-item]
            "rope_type": "default",  # type: ignore[dict-item]
            "rope_theta": 5_000_000,  # type: ignore[dict-item]
        },
        tie_word_embeddings=True,  # type: ignore[call-arg]
        use_cache=True,
        vocab_size=151936,
    )
    vision_config = Qwen3VLVisionConfig(
        deepstack_visual_indexes=[5, 11, 17],
        depth=24,
        hidden_act="gelu_pytorch_tanh",
        hidden_size=1024,
        in_channels=3,
        initializer_range=0.02,
        intermediate_size=4096,
        num_heads=16,
        num_position_embeddings=2304,
        out_hidden_size=2048,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    return Qwen3VLConfig(
        architectures=["Qwen3VLForConditionalGeneration"],
        image_token_id=151655,
        text_config=text_config.to_dict(),
        tie_word_embeddings=True,
        video_token_id=151656,
        vision_config=vision_config.to_dict(),
        vision_end_token_id=151653,
        vision_start_token_id=151652,
    )
