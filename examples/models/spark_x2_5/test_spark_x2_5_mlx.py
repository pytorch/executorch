"""
Numerical tests for Spark-X2.5 MLX attention.

Tests that MLXAttentionMHA produces the same output as eager AttentionMHA
for Spark-X2.5 configurations, including:
- Sliding window attention with ring buffer cache
- Headwise output gate
- Per-layer RoPE parameters (partial rotary factor)
- Ring buffer sizing, chunked prefill, and the single-write prefill bound
"""

import pytest
import torch
from executorch.backends.mlx.llm.cache import KVCache, RingBufferKVCache
from executorch.backends.mlx.llm.et_attention import MLXAttentionMHA
from executorch.examples.models.llama.attention import AttentionMHA
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope


def _create_tiny_model_args(
    layer_types: list[str],
    sliding_window: int = 4,
    headwise_attn_output_gate: bool = True,
    max_seq_len: int = 8,
    max_context_len: int = 32,
    num_kv_shared_layers: int = 0,
) -> ModelArgs:
    """Create tiny ModelArgs for testing Spark-X2.5 attention."""
    return ModelArgs(
        dim=64,
        n_layers=max(len(layer_types), num_kv_shared_layers + 1),
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        hidden_dim=128,
        vocab_size=100,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        max_context_len=max_context_len,
        use_kv_cache=True,
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
        num_kv_shared_layers=num_kv_shared_layers,
        # Per-layer RoPE parameters
        rope_parameters={
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
        },
    )


def _create_layer_args(args: ModelArgs, layer_type: str) -> ModelArgs:
    """Copy args with per-layer-type RoPE parameters applied."""
    rope_params = args.rope_parameters.get(layer_type, {})
    args_dict = {**args.__dict__}
    # Set both rope_theta and rope_freq_base to ensure __post_init__ doesn't override
    layer_rope_theta = rope_params.get("rope_theta", args.rope_theta)
    args_dict["rope_theta"] = layer_rope_theta
    args_dict["rope_freq_base"] = layer_rope_theta
    args_dict["partial_rotary_factor"] = rope_params.get("partial_rotary_factor", 1.0)
    return ModelArgs(**args_dict)


def _create_eager_attention(
    args: ModelArgs, layer_id: int, layer_type: str
) -> AttentionMHA:
    """Create an eager AttentionMHA for a specific layer."""
    # Temporarily set attention_type to default for eager model
    args_copy = ModelArgs(**{**args.__dict__, "attention_type": "default"})
    return AttentionMHA(args_copy, layer_id, Rope(_create_layer_args(args, layer_type)))


def _run_both(
    eager_attn: AttentionMHA,
    mlx_attn: MLXAttentionMHA,
    x: torch.Tensor,
    start_pos: int,
) -> None:
    """Run one forward step on both modules and assert outputs match."""
    seq_len = x.shape[1]
    input_pos = torch.arange(start_pos, start_pos + seq_len)
    freqs_cos, freqs_sin = eager_attn.rope.get_freqs(input_pos, seq_len)
    eager_out, _ = eager_attn(
        x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, input_pos=input_pos
    )
    mlx_out, _ = mlx_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)
    torch.testing.assert_close(eager_out, mlx_out, rtol=1e-4, atol=1e-5)


def test_mlx_attention_matches_eager_sliding_window() -> None:
    """Test that MLX attention matches eager attention for sliding window layers."""
    torch.manual_seed(42)

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

    x = torch.randn(1, 2, args.dim)
    _run_both(eager_attn, mlx_attn, x, start_pos=0)


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

    # Decode one token at a time, past the window of 4
    for pos in range(8):
        x = torch.randn(1, 1, args.dim)
        _run_both(eager_attn, mlx_attn, x, start_pos=pos)


def test_mlx_attention_chunked_prefill_past_window() -> None:
    """Test multi-token prefill chunks that cross the sliding window boundary."""
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

    # Chunks of <= max_write_len (min(max_seq_len, window) = 4); the second
    # chunk crosses the window boundary and the third evicts positions.
    pos = 0
    for chunk_len in (3, 3, 2, 1):
        x = torch.randn(1, chunk_len, args.dim)
        _run_both(eager_attn, mlx_attn, x, start_pos=pos)
        pos += chunk_len


def test_mlx_attention_prefill_longer_than_window_raises() -> None:
    """A single forward writing more tokens than max_write_len must fail loudly.

    The ring buffer cannot keep a contiguous in-window range for writes longer
    than max_write_len, so long prompts must be chunked (see README).
    """
    torch.manual_seed(42)

    layer_types = ["sliding_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=4,
        headwise_attn_output_gate=True,
        max_seq_len=8,
    )

    eager_attn = _create_eager_attention(
        args, layer_id=0, layer_type="sliding_attention"
    )
    mlx_attn = MLXAttentionMHA.from_attention_mha(eager_attn, dtype=torch.float32)
    assert mlx_attn.kv_cache.max_write_len == 4

    x = torch.randn(1, 8, args.dim)
    input_pos = torch.arange(8)
    with pytest.raises(RuntimeError, match="max_write_len"):
        mlx_attn(x, freqs_cos=None, freqs_sin=None, input_pos=input_pos)


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

    x = torch.randn(1, 2, args.dim)
    _run_both(eager_attn, mlx_attn, x, start_pos=0)


def test_ring_buffer_sized_by_max_seq_len() -> None:
    """The ring's max_write_len must come from max_seq_len, not max_context_len.

    max_seq_len bounds the prefill chunk; sizing by max_context_len would
    over-allocate the ring slack whenever the context is longer than the
    largest single prefill.
    """
    torch.manual_seed(42)

    layer_types = ["sliding_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=16,
        max_seq_len=8,
        max_context_len=32,
    )

    eager_attn = _create_eager_attention(
        args, layer_id=0, layer_type="sliding_attention"
    )
    mlx_attn = MLXAttentionMHA.from_attention_mha(eager_attn, dtype=torch.float32)

    assert isinstance(mlx_attn.kv_cache, RingBufferKVCache)
    assert mlx_attn.kv_cache.max_write_len == min(args.max_seq_len, args.sliding_window)
    assert mlx_attn.kv_cache.buffer_size == args.sliding_window + 8 - 1


def test_plain_cache_when_context_within_window() -> None:
    """When max_context_len <= sliding_window the window never evicts, so a
    plain KVCache is used instead of an oversized ring buffer."""
    torch.manual_seed(42)

    layer_types = ["sliding_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=64,
        max_seq_len=8,
        max_context_len=32,
    )

    eager_attn = _create_eager_attention(
        args, layer_id=0, layer_type="sliding_attention"
    )
    mlx_attn = MLXAttentionMHA.from_attention_mha(eager_attn, dtype=torch.float32)

    assert not isinstance(mlx_attn.kv_cache, RingBufferKVCache)
    assert isinstance(mlx_attn.kv_cache, KVCache)
    assert mlx_attn.kv_cache.k_cache.shape[2] == args.max_context_len

    # Numerics still match eager (the window never binds within 32 tokens)
    _run_both(eager_attn, mlx_attn, torch.randn(1, 3, args.dim), start_pos=0)
    _run_both(eager_attn, mlx_attn, torch.randn(1, 1, args.dim), start_pos=3)


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

    eager_attn.use_attn_o_norm = True
    with pytest.raises(NotImplementedError, match="use_attn_o_norm"):
        MLXAttentionMHA.from_attention_mha(eager_attn)
    eager_attn.use_attn_o_norm = False

    eager_attn.scale_query_by = 2.0
    with pytest.raises(NotImplementedError, match="scale_query_by"):
        MLXAttentionMHA.from_attention_mha(eager_attn)
    eager_attn.scale_query_by = 1.0

    eager_attn.use_attn_o_gate = True
    eager_attn.headwise_attn_output_gate = False
    with pytest.raises(NotImplementedError, match="use_attn_o_gate"):
        MLXAttentionMHA.from_attention_mha(eager_attn)
    eager_attn.use_attn_o_gate = False
    eager_attn.headwise_attn_output_gate = True

    eager_attn.use_q_gate = True
    with pytest.raises(NotImplementedError, match="use_q_gate"):
        MLXAttentionMHA.from_attention_mha(eager_attn)
    eager_attn.use_q_gate = False

    eager_attn.is_kv_shared_layer = True
    with pytest.raises(NotImplementedError, match="is_kv_shared_layer"):
        MLXAttentionMHA.from_attention_mha(eager_attn)


def test_mlx_attention_init_rejects_kv_shared_layers() -> None:
    """MLXAttentionMHA.__init__ must reject KV-shared (YOCO) layers.

    ModelArgs has num_kv_shared_layers, not a per-layer is_kv_shared_layer
    flag, so the check has to derive the flag from layer_id.
    """
    torch.manual_seed(42)

    layer_types = ["full_attention", "sliding_attention"]
    args = _create_tiny_model_args(
        layer_types=layer_types,
        sliding_window=4,
        num_kv_shared_layers=1,
    )
    rope_args = _create_layer_args(args, "sliding_attention")

    # layer_id 1 of n_layers 2 with num_kv_shared_layers=1 is a shared layer
    with pytest.raises(NotImplementedError, match="is_kv_shared_layer"):
        MLXAttentionMHA(args, layer_id=1, rope=Rope(rope_args))

    # layer_id 0 is not shared and constructs fine
    MLXAttentionMHA(args, layer_id=0, rope=Rope(rope_args))
