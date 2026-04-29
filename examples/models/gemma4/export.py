"""Export Gemma4-31B (text) to ExecuTorch .pte + .ptd via the CUDA backend.

Two methods are emitted:
  - decode  : T=1, static shape
  - prefill : T>=2, dynamic seq_len up to max_seq_len-1
Both methods share the model's KV-cache buffers via share_mutable_buffers=True.

Usage:
  python export.py --tiny-test --output-dir /tmp/gemma4_tiny
  python export.py --model-dir /path/to/gemma-4-31B --output-dir /tmp/gemma4 \\
                   --max-seq-len 4096
"""

import argparse
import os

import torch
import torch.nn as nn

from executorch.examples.models.gemma4.model import (
    Gemma4TextConfig,
    Gemma4TextModel,
)


# Tiny config exercises the same architecture (sliding+full mix, k_eq_v,
# partial RoPE, layer_scalar, softcapping) at toy dimensions for CI.
def build_tiny_config() -> Gemma4TextConfig:
    layer_types = [
        "sliding_attention" if (i + 1) % 3 != 0 else "full_attention"
        for i in range(6)
    ]
    return Gemma4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=32,
        global_head_dim=64,
        sliding_window=16,
        rms_norm_eps=1e-6,
        sliding_rope_theta=10_000.0,
        full_rope_theta=1_000_000.0,
        full_partial_rotary_factor=0.25,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        pad_token_id=0,
        max_seq_len=64,
        layer_types=layer_types,
    )


def build_tiny_model(seed: int = 42) -> tuple[Gemma4TextModel, Gemma4TextConfig]:
    config = build_tiny_config()
    torch.manual_seed(seed)
    model = Gemma4TextModel(config)
    model.to(dtype=torch.bfloat16)
    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def load_full_model(
    model_dir: str, max_seq_len: int
) -> tuple[Gemma4TextModel, Gemma4TextConfig]:
    print(f"Loading Gemma4 from {model_dir}...")
    model, config = Gemma4TextModel.from_hf_checkpoint(
        model_dir, max_seq_len=max_seq_len
    )
    model.to(dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def load_prequantized_model(
    prequantized_dir: str, max_seq_len: int
) -> tuple[Gemma4TextModel, Gemma4TextConfig]:
    """Load a prequantized Gemma4 bundle produced by quantize_and_save.py.

    Mirrors qwen3_5_moe.export.load_prequantized_model: build on meta device,
    then assign tensors from the safetensors bundle (subclass tensors are
    reconstructed by load_quantized_state_dict).
    """
    from executorch.examples.models.gemma4.model import Gemma4TextConfig
    from executorch.examples.models.qwen3_5_moe.quantize_and_save import (
        load_quantized_state_dict,
    )

    config_path = os.path.join(prequantized_dir, "config.json")
    safetensors_path = os.path.join(prequantized_dir, "model.safetensors")

    config = Gemma4TextConfig.from_hf_config(config_path)
    config.max_seq_len = max_seq_len

    print(f"Loading prequantized weights from {safetensors_path}...")
    state_dict = load_quantized_state_dict(safetensors_path)

    print("Building model on meta device...")
    with torch.device("meta"):
        model = Gemma4TextModel(config)

    missing, unexpected = model.load_state_dict(
        state_dict, strict=False, assign=True
    )
    del state_dict

    # Re-tie lm_head <-> embed_tokens. The bundle only contains
    # embed_tokens.weight (named_parameters() dedupes tied params), and
    # assign=True replaces embed_tokens.weight with a new Parameter object
    # while leaving the meta-device lm_head.weight orphaned.
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    runtime_prefixes = (
        ".k_cache",
        ".v_cache",
        ".attn_mask",
        ".inv_freq",
        ".layer_scalar",
    )
    weight_missing = [
        k for k in missing
        if not any(p in k for p in runtime_prefixes)
        and ".v_proj." not in k
        and k != "lm_head.weight"  # tied to embed_tokens
    ]
    if weight_missing:
        print(f"WARNING: prequant missing weights: {weight_missing[:8]}")
    if unexpected:
        print(f"WARNING: prequant unexpected keys: {sorted(unexpected)[:8]}")

    # load_state_dict(assign=True) re-wraps tensors as Parameter(requires_grad=True);
    # quantized subclass inner tensors (int dtype) can't be Parameters with grad.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, config


def materialize_buffers(model: Gemma4TextModel, config: Gemma4TextConfig) -> None:
    """Replace any meta-device buffers with real tensors and recompute RoPE/masks.

    Buffers are not loaded from the checkpoint (KV cache, attn_mask, inv_freq,
    layer_scalar) so a meta-device build leaves them on `meta`. Reinit on CPU.
    """
    for fqn, buf in list(model.named_buffers()):
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        if buf.device.type == "meta":
            dtype = torch.bfloat16 if buf.dtype.is_floating_point else buf.dtype
            parent.register_buffer(
                parts[-1],
                torch.zeros(buf.shape, dtype=dtype, device="cpu"),
            )

    for layer in model.layers:
        attn = layer.self_attn
        rotary_dim = attn.rotary_dim
        rope_theta = (
            config.sliding_rope_theta if attn.is_sliding else config.full_rope_theta
        )
        attn.inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(config.max_seq_len)
        causal = positions[None, :] <= positions[:, None]
        if attn.is_sliding:
            within_window = (positions[:, None] - positions[None, :]) < config.sliding_window
            mask = causal & within_window
        else:
            mask = causal
        attn.register_buffer("attn_mask", mask)
        # KV caches stay zero-initialized. Now wrapped in attn.kv_cache submodule.
        attn.kv_cache.register_buffer(
            "k_cache",
            torch.zeros(1, attn.num_kv_heads, config.max_seq_len, attn.head_dim, dtype=torch.bfloat16),
        )
        attn.kv_cache.register_buffer(
            "v_cache",
            torch.zeros(1, attn.num_kv_heads, config.max_seq_len, attn.head_dim, dtype=torch.bfloat16),
        )
        # layer_scalar is a LEARNED parameter in the HF checkpoint — only init
        # to 1.0 when it's still on meta (no checkpoint loaded).
        if layer.layer_scalar.device.type == "meta":
            layer.register_buffer("layer_scalar", torch.ones(1, dtype=torch.bfloat16))


def export_two_methods(model: Gemma4TextModel, config: Gemma4TextConfig):
    """Run torch.export twice (decode T=1, prefill T>=2). Returns (decode_ep, prefill_ep).

    Example inputs are placed on the same device as the model's parameters so
    tracing works whether the model is on CPU or CUDA.
    """
    from torch.export import Dim, export

    try:
        device = next(p for p in model.parameters() if p.device.type != "meta").device
    except StopIteration:
        device = torch.device("cpu")

    decode_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
    decode_pos = torch.tensor([0], dtype=torch.long, device=device)
    print("torch.export: decode (T=1)...")
    with torch.no_grad():
        decode_ep = export(model, (decode_tokens, decode_pos), strict=True)
    print("  decode export OK")

    max_prefill = config.max_seq_len - 1
    prefill_tokens = torch.zeros((1, max_prefill), dtype=torch.long, device=device)
    prefill_pos = torch.arange(max_prefill, dtype=torch.long, device=device)
    seq_dim = Dim("seq_len", min=2, max=max_prefill)
    dynamic_shapes = ({1: seq_dim}, {0: seq_dim})
    print(f"torch.export: prefill (T<={max_prefill})...")
    with torch.no_grad():
        prefill_ep = export(
            model,
            (prefill_tokens, prefill_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print("  prefill export OK")
    return decode_ep, prefill_ep


def lower_and_save(decode_ep, prefill_ep, config: Gemma4TextConfig, output_dir: str):
    """to_edge_transform_and_lower with CUDA backend; emit .pte + .ptd."""
    import torch._inductor.config as inductor_config

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass

    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"
    # Disable max-autotune to avoid OOM during AOTI: the benchmark step
    # allocates large example tensors (~2.6 GB each) that push us past
    # the 80 GB budget on top of the 16-18 GB INT4 model + KV caches.
    inductor_config.max_autotune = False
    inductor_config.max_autotune_gemm = False

    metadata = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }

    print("to_edge_transform_and_lower (CUDA)...")
    et_prog = to_edge_transform_and_lower(
        {"decode": decode_ep, "prefill": prefill_ep},
        partitioner={
            "decode": [
                CudaPartitioner(
                    [
                        CudaBackend.generate_method_name_compile_spec("decode"),
                    ]
                )
            ],
            "prefill": [
                CudaPartitioner(
                    [
                        CudaBackend.generate_method_name_compile_spec("prefill"),
                    ]
                )
            ],
        },
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )
    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                share_mutable_buffers=True,
            ),
            emit_mutable_buffer_names=True,
        ),
    )

    os.makedirs(output_dir, exist_ok=True)
    pte_path = os.path.join(output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"  saved {size_mb:.1f} MB")

    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)
        print(f"  saved tensor data to {output_dir}/")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Export Gemma4 text model to ExecuTorch")
    parser.add_argument("--model-dir", default=None,
                        help="HuggingFace gemma-4 checkpoint directory (bf16)")
    parser.add_argument("--prequantized", default=None,
                        help="Prequantized Gemma4 bundle from quantize_and_save.py")
    parser.add_argument("--output-dir", default="./gemma4_export",
                        help="Output directory for .pte/.ptd files")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="KV cache length (default 4096)")
    parser.add_argument("--tiny-test", action="store_true",
                        help="Build a tiny random-weight model for CI pipeline check")
    args = parser.parse_args()

    if not args.tiny_test and not args.model_dir and not args.prequantized:
        parser.error("--tiny-test, --model-dir, or --prequantized is required")

    if args.tiny_test:
        model, config = build_tiny_model()
    elif args.prequantized:
        model, config = load_prequantized_model(args.prequantized, args.max_seq_len)
    else:
        model, config = load_full_model(args.model_dir, args.max_seq_len)

    materialize_buffers(model, config)

    decode_ep, prefill_ep = export_two_methods(model, config)
    lower_and_save(decode_ep, prefill_ep, config, args.output_dir)


if __name__ == "__main__":
    main()
