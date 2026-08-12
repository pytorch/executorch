# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parity test comparing DFlashQwen3Attention with unmodified Qwen3Attention.

With an empty target context, DFlash should match bidirectional self-attention
over the proposal block. Any mismatch indicates a difference in the adapted
attention logic, such as position handling, masking, or dispatch."""

import copy

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_model import (
    _to_qwen3_config,
    DFlashConfig,
    DFlashQwen3Attention,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
)


def main():
    torch.manual_seed(0)

    config = DFlashConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=100,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        target_layer_ids=(0,),
    )
    qwen3_config = _to_qwen3_config(config)
    qwen3_config._attn_implementation = "eager"

    dflash_attn = DFlashQwen3Attention(qwen3_config, layer_idx=0).eval()
    ref_attn = Qwen3Attention(qwen3_config, layer_idx=0).eval()
    ref_attn.is_causal = False

    ref_attn.load_state_dict(copy.deepcopy(dflash_attn.state_dict()))

    B, L = 1, 5
    x = torch.randn(B, L, config.hidden_size)
    x_ctx = torch.zeros(B, 0, config.hidden_size)

    rotary_emb = Qwen3RotaryEmbedding(qwen3_config)
    position_ids = torch.arange(L).unsqueeze(0)
    cos, sin = rotary_emb(x, position_ids)

    dflash_out = dflash_attn(x, x_ctx, cos, sin)
    ref_out, _ = ref_attn(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
    )

    max_diff = (dflash_out - ref_out).abs().max().item()
    print(f"max abs diff: {max_diff:.3e}")
    if torch.allclose(dflash_out, ref_out, atol=1e-5, rtol=1e-5):
        print(
            "PASS: DFlashQwen3Attention matches real Qwen3Attention with empty context."
        )
    else:
        print("FAIL: DFlashQwen3Attention diverges from the reference module.")


if __name__ == "__main__":
    main()
