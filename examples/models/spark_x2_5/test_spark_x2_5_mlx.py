"""
Numerical tests for Spark-X2.5 MLX attention.

Tests that MLXAttentionMHA produces the same output as eager AttentionMHA
for Spark-X2.5 configurations, including:
- Sliding window attention with ring buffer cache
- Headwise output gate
- Per-layer RoPE parameters (partial rotary factor)
"""

import torch
from executorch.backends.mlx.llm.et_attention import MLXAttentionMHA
from executorch.examples.models.llama.attention import AttentionMHA
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope


def _create_tiny_model_args(
    layer_types: list[str],
    sliding_window: int = 4,
    headwise_attn_output_gate: bool = True,
) -> ModelArgs:
    """Create tiny ModelArgs for testing Spark-X2.5 attention."""
    return ModelArgs(
        dim=64,
        n_layers=len(layer_types),
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        hidden_dim=128,
        vocab_size=100,
        max_batch_size=1,
        max_seq_len=32,
        max_context_len=32,
        use_kv_cache=True,
        use_sdpa_with_kv_cache=True,
        norm_eps=1e-5,
        rope_theta=10000.0,
        use_hf_rope=True,
        use_scaled_rope=False,
        sliding_window=sliding_window,
        layer_types=layer_types,
        headwise_attn_output_gate=headwise_attn_output_gate,
        attention_qkv_bias=False,
        use_qk_norm=False,
        qk_norm_before_rope=False,
        act_fn="gelu",
        attention_type="mlx",
        # Per-layer RoPE parameters
        rope_parameters={
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
        },
    )


def _create_eager_attention(
    args: ModelArgs, layer_id: int, layer_type: str
) -> AttentionMHA:
    """Create an eager AttentionMHA for a specific layer."""
    # Temporarily set attention_type to default for eager model
    args_copy = ModelArgs(**{**args.__dict__, "attention_type": "default"})

    # Create rope params with layer-specific parameters
    rope_params = args.rope_parameters.get(layer_type, {})
    rope_args_dict = {**args.__dict__}
    rope_args_dict["rope_freq_base"] = rope_params.get("rope_theta", args.rope_theta)
    rope_args_dict["partial_rotary_factor"] = rope_params.get(
        "partial_rotary_factor", 1.0
    )
    rope_args = ModelArgs(**rope_args_dict)
    rope = Rope(rope_args)

    return AttentionMHA(args_copy, layer_id, rope)


def test_mlx_attention_matches_eager_sliding_window() -> None:
    """Test that MLX attention matches eager attention for sliding window layers."""
    torch.manual_seed(42)

    # Create a tiny model with sliding window
    layer_types = ["sliding_attention", "full_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=4,
        headwise_attn_output_gate=True,
    )

    # Test sliding attention layer (layer 0)
    eager_attn = _create_eager_attention(
        args, layer_id=0, layer_type="sliding_attention"
    )
    mlx_attn = MLXAttentionMHA.from_attention_mha(eager_attn, dtype=torch.float32)

    # Verify sliding window is correctly set
    assert mlx_attn.is_sliding is True
    assert mlx_attn.sliding_window == 4
    assert mlx_attn.headwise_attn_output_gate is True

    # Create input
    batch_size = 1
    seq_len = 2
    x = torch.randn(batch_size, seq_len, args.dim)

    # Create position indices
    input_pos = torch.tensor([0, 1])

    # Run eager attention
    eager_out, _ = eager_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)

    # Run MLX attention
    mlx_out, _ = mlx_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)

    # Compare outputs (should match within numerical precision)
    torch.testing.assert_close(eager_out, mlx_out, rtol=1e-4, atol=1e-5)


def test_mlx_attention_matches_eager_past_window() -> None:
    """Test that MLX attention correctly handles positions past the sliding window."""
    torch.manual_seed(42)

    layer_types = ["sliding_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=4,
        headwise_attn_output_gate=True,
    )

    eager_attn = _create_eager_attention(
        args, layer_id=0, layer_type="sliding_attention"
    )
    mlx_attn = MLXAttentionMHA.from_attention_mha(eager_attn, dtype=torch.float32)

    # Decode past the window size
    batch_size = 1
    seq_len = 1

    # Prefill positions 0-5 to populate KV cache
    for pos in range(6):
        input_pos = torch.tensor([pos])
        x = torch.randn(batch_size, seq_len, args.dim)
        eager_out, _ = eager_attn(
            x, freqs_cos=None, freqs_sin=None, input_pos=input_pos
        )
        mlx_out, _ = mlx_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)
        # Verify each step matches
        torch.testing.assert_close(eager_out, mlx_out, rtol=1e-4, atol=1e-5)

    # Now test at position 6 (past the window of 4)
    input_pos = torch.tensor([6])
    x = torch.randn(batch_size, seq_len, args.dim)

    # Run both
    eager_out, _ = eager_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)
    mlx_out, _ = mlx_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)

    # Should still match
    torch.testing.assert_close(eager_out, mlx_out, rtol=1e-4, atol=1e-5)


def test_mlx_attention_full_attention_layer() -> None:
    """Test MLX attention for full attention layer with partial RoPE."""
    torch.manual_seed(42)

    layer_types = ["full_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=4,
        headwise_attn_output_gate=True,
    )

    eager_attn = _create_eager_attention(args, layer_id=0, layer_type="full_attention")
    mlx_attn = MLXAttentionMHA.from_attention_mha(eager_attn, dtype=torch.float32)

    # Full attention should not be sliding
    assert mlx_attn.is_sliding is False
    assert mlx_attn.sliding_window is None

    # Verify partial rotary factor is set correctly
    assert mlx_attn.rope_dims == int(args.head_dim * 0.25)  # partial_rotary_factor=0.25

    # Test forward pass
    batch_size = 1
    seq_len = 2
    x = torch.randn(batch_size, seq_len, args.dim)
    input_pos = torch.tensor([0, 1])

    eager_out, _ = eager_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)
    mlx_out, _ = mlx_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)

    torch.testing.assert_close(eager_out, mlx_out, rtol=1e-4, atol=1e-5)


def test_mlx_attention_unsupported_options() -> None:
    """Test that unsupported options raise errors."""
    torch.manual_seed(42)

    layer_types = ["sliding_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=4,
        headwise_attn_output_gate=True,
    )

    eager_attn = _create_eager_attention(
        args, layer_id=0, layer_type="sliding_attention"
    )

    # Test use_attn_o_norm (unsupported)
    eager_attn.use_attn_o_norm = True
    try:
        MLXAttentionMHA.from_attention_mha(eager_attn)
        assert False, "Should have raised NotImplementedError for use_attn_o_norm"
    except NotImplementedError as e:
        assert "use_attn_o_norm" in str(e)

    # Reset and test scale_query_by (unsupported)
    eager_attn.use_attn_o_norm = False
    eager_attn.scale_query_by = 2.0
    try:
        MLXAttentionMHA.from_attention_mha(eager_attn)
        assert False, "Should have raised NotImplementedError for scale_query_by"
    except NotImplementedError as e:
        assert "scale_query_by" in str(e)
