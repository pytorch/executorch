import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 1
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 512  # Arbitrary value, should be defined later by tokenizer.
    hidden_dim: Optional[int] = None
    head_dim: Optional[int] = None  # Optional customized head_dim
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 1
    max_seq_len: int = 2048
    max_context_len: int = 2048
    moe: bool = False  # True to enable the MoE (Mixture of Experts)
    num_experts: int = 8  # Number of experts
    num_activated_experts: int = 2  # Number of experts to activate
    attention_type: str = "mha"  # Attention type, registered in attention.py
    attention_qkv_bias: bool = False
    use_kv_cache: bool = False  # Use key/value cache
    use_sdpa_with_kv_cache_op: bool = (
        False  # Use custom sdpa op that updates kv cache in-place
    )
    # Generate logits for all inputs. When it's True, it would take big memory usage
    # at runtime. Enable it only necessary (e.g., use perplexity tools that requires
    # logits for all input tokens.)
    generate_full_logits: bool = False
    enable_dynamic_shape: bool = False  # export model with dynamic shape support
    # A dictionary mapping from pruned token-id to original token-id
    input_prune_map: Optional[Dict[int, int]] = None
    # A dictionary mapping from pruned token-id to original token-id
    output_prune_map: Optional[Dict[int, int]] = None
    apply_embedding: bool = True  # Use embedding inside the transformer
    apply_output: bool = True  # Use output layer (unembedding) inside the transformer
    use_qk_norm: bool = False  # apply normalization to q and k in the attention
    qk_norm_before_rope: bool = False  # when to apply qk norm
    use_hf_rope: bool = False  # Use HuggingFace's RoPE implementation
    partial_rotary_factor: float = 1.0
    rope_theta: Optional[float] = (
        None  # The official name to override self.rope_freq_base.
    )
    rope_freq_base: float = 10000.0  # The base frequency for RoPE. Keep it for BC.
    use_scaled_rope: bool = False  # Use scaled RoPE, introduced in llama3.1.
    rope_scale_factor: int = 8
    high_freq_factor: int = 4
    # Additional Model Metadata needed at runtime
    bos_idx: int = 1
    eos_idx: int = 3
    bos_count: int = -1  # i.e., a single EOS is used as BOS
    eos_count: int = 2

    quantization_args: Optional[dict] = None
    # LoRA for QAT.
    lora_args: Optional[dict] = None

    # LoRA arguments to set up a LoRA inference model.
    # These arguments come directly from a torchtune adapter_config.json file.
    r: Optional[int] = None  # Rank.
    lora_alpha: Optional[int] = None  # Alpha.
    # Eg. q_proj, k_proj, v_proj, output_proj
    target_modules: Optional[list] = None
    peft_type: Optional[str] = None  # PEFT type.
    base_model_name_or_path: Optional[str] = None  # Base model name or path.
    kv_io_bit_width: Optional[int] = (
        None  # KV cache bit width. This is for QNN backend only for now.
    )
    attention_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Hybrid models can have layer types different from attention
    layer_types: Optional[list] = None

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # rope_theta overrides rope_freq_base since it's the official name.
        if self.rope_theta is not None:
            self.rope_freq_base = self.rope_theta

        if self.use_sdpa_with_kv_cache_op:
            assert self.use_kv_cache, "use_sdpa_with_kv_cache_op requires use_kv_cache"

        if self.hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = self.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)

            def find_multiple(n: int, k: int) -> int:
                if n % k == 0:
                    return n
                return n + k - (n % k)

            self.hidden_dim = find_multiple(hidden_dim, multiple_of)

        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
