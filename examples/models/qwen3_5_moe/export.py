"""
Export Qwen 3.5 MoE to ExecuTorch .pte format (CUDA only).

Usage:
  python export.py --model-dir /path/to/Qwen3.5-MoE-A3B
  python export.py --model-dir /path/to/model --qlinear 4w
  python export.py --prequantized /path/to/quantized_bundle/
"""

import argparse
import os

import torch
import torch.nn as nn

from executorch.examples.models.qwen3_5_moe.model import (
    FusedMoEExperts,
    Qwen35MoE,
    Qwen35MoEConfig,
)


# ---------------------------------------------------------------------------
# Load + quantize
# ---------------------------------------------------------------------------


def load_and_quantize(args):
    """Load model from checkpoint, optionally quantize, move to CUDA.

    Returns (model, config) ready for export.
    """
    if args.prequantized:
        return load_prequantized_model(args.prequantized, args.max_seq_len)

    print("Loading model...")
    model, config = Qwen35MoE.from_hf_checkpoint(
        args.model_dir, max_seq_len=args.max_seq_len
    )
    model.eval()
    print(
        f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
        f"{config.num_experts} experts top-{config.num_experts_per_tok}"
    )

    if args.qlinear or args.qembedding:
        _quantize(model, config, args)
    else:
        model.to(dtype=torch.bfloat16)

    return model, config


def load_prequantized_model(prequantized_dir, max_seq_len=4096):
    """Load a prequantized safetensors bundle into a model.

    Args:
        prequantized_dir: Directory containing model.safetensors and config.json.
        max_seq_len: Maximum sequence length for KV cache.

    Returns:
        (model, config) ready for export.
    """
    from executorch.examples.models.qwen3_5_moe.quantize_and_save import (
        load_quantized_state_dict,
    )

    config_path = os.path.join(prequantized_dir, "config.json")
    safetensors_path = os.path.join(prequantized_dir, "model.safetensors")

    config = Qwen35MoEConfig.from_hf_config(config_path)
    config.max_seq_len = max_seq_len

    print(f"Loading prequantized weights from {safetensors_path}...")
    state_dict = load_quantized_state_dict(safetensors_path)

    # Build model on meta device and prepare for quantized expert buffers.
    # The model init creates w1_weight/w2_weight parameters but the checkpoint
    # has w1/w1_scale/w2/w2_scale buffers. Replace them with matching placeholders
    # so load_state_dict can assign the quantized weights.
    print("Building model on meta device...")
    with torch.device("meta"):
        model = Qwen35MoE(config)

    for i, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        if isinstance(experts, FusedMoEExperts) and hasattr(experts, "w1_weight"):
            del experts.w1_weight
            del experts.w2_weight
            prefix = f"layers.{i}.mlp.experts"
            for buf_name in ("w1", "w1_scale", "w2", "w2_scale"):
                t = state_dict[f"{prefix}.{buf_name}"]
                experts.register_buffer(
                    buf_name,
                    torch.empty(t.shape, dtype=t.dtype, device="meta"),
                )
            # Infer group_size from packed weight and scale shapes:
            # w1 is [E, N, K//2] (packed int4), w1_scale is [E, N, K//gs]
            w1 = state_dict[f"{prefix}.w1"]
            w1_scale = state_dict[f"{prefix}.w1_scale"]
            experts.group_size = (w1.shape[2] * 2) // w1_scale.shape[2]

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    # Validate: only runtime state buffers should be missing.
    # Any missing weight key indicates a version mismatch between the
    # checkpoint and the model (e.g., unfused vs fused projections).
    runtime_prefixes = (
        ".kv_cache.",
        ".conv_state",
        ".recurrent_state",
        ".cache_positions",
        ".inv_freq",
    )
    expected_missing = {k for k in missing if any(p in k for p in runtime_prefixes)}
    weight_missing = set(missing) - expected_missing
    if weight_missing:
        raise RuntimeError(
            f"Prequantized checkpoint is missing {len(weight_missing)} weight keys "
            f"(model/checkpoint version mismatch?): {sorted(weight_missing)[:10]}"
        )
    if unexpected:
        raise RuntimeError(
            f"Prequantized checkpoint has {len(unexpected)} unexpected keys "
            f"(model/checkpoint version mismatch?): {sorted(unexpected)[:10]}"
        )

    # load_state_dict(assign=True) wraps tensors as Parameter(requires_grad=True).
    # run_decompositions -> unwrap_tensor_subclass_parameters tries to wrap
    # int-dtype inner tensors of quantized subclasses as Parameters with
    # requires_grad=True, which fails. Disable grad on all parameters.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    print(
        f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
        f"{config.num_experts} experts top-{config.num_experts_per_tok}"
    )
    return model, config


def _quantize_experts_int4(model, config, group_size=32, use_hqq=False):
    """Quantize expert weights to packed INT4 for the fused MoE kernel.

    Two quantization methods:
      --hqq: HQQ (Half-Quadratic Quantization) iteratively refines scales
             via least-squares for better accuracy (slower).
      default: Standard min/max symmetric quantization (faster).

    Converts w1_weight [E, N, K] and w2_weight [E, N, K] to:
      w1 [E, N, K//2] int8 packed, w1_scale [E, N, K//gs] bf16
      w2 [E, N, K//2] int8 packed, w2_scale [E, N, K//gs] bf16
    """
    if use_hqq:
        from torchao.quantization.quant_primitives import (
            _choose_qparams_and_quantize_scale_only_hqq,
        )
    else:
        from torchao.quantization.quant_primitives import (
            choose_qparams_affine,
            MappingType,
            quantize_affine,
        )

    method = "HQQ" if use_hqq else "min/max"

    for i, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        if not isinstance(experts, FusedMoEExperts):
            continue

        experts.group_size = group_size
        for name in ("w1_weight", "w2_weight"):
            w = getattr(experts, name).data.float()
            E, N, K = w.shape

            if use_hqq:
                qdata, scale = _choose_qparams_and_quantize_scale_only_hqq(
                    w.view(E * N, K),
                    block_size=[1, group_size],
                    qmin=-8,
                    qmax=7,
                )
                int_data = qdata.to(torch.int8).view(E, N, K)
                scale = scale.view(E, N, -1)
            else:
                block_size = (1, 1, group_size)
                scale, zero_point = choose_qparams_affine(
                    w,
                    MappingType.SYMMETRIC,
                    block_size,
                    target_dtype=torch.int8,
                    quant_min=-8,
                    quant_max=7,
                )
                int_data = quantize_affine(
                    w,
                    block_size,
                    scale,
                    zero_point,
                    output_dtype=torch.int8,
                    quant_min=-8,
                    quant_max=7,
                )
                scale = scale.reshape(E, N, -1)

            # Pack two int4 values per byte: even K -> low nibble, odd K -> high nibble
            uint4 = (int_data + 8).to(torch.int16)  # shift to unsigned [0, 15]
            low = uint4[:, :, 0::2]
            high = uint4[:, :, 1::2]
            packed = (low | (high << 4)).to(torch.int8)  # [E, N, K//2]

            buf_name = name.replace("_weight", "")
            experts.register_buffer(buf_name, packed)
            experts.register_buffer(f"{buf_name}_scale", scale.to(torch.bfloat16))
            delattr(experts, name)

        print(
            f"  Quantized experts (INT4 {method}) layer {i + 1}/{config.num_hidden_layers}",
            end="\r",
        )
    print()


def _to_device_skip_meta(module, device, dtype=None):
    """Move submodules to device, skipping any that have meta-device buffers.

    Uses module.to() on leaf submodules (not p.data = p.data.to()) to
    correctly handle tensor subclasses like Int4TilePackedTo4dTensor.
    """
    for _, submod in module.named_modules():
        has_meta = any(
            b.device.type == "meta" for _, b in submod.named_buffers(recurse=False)
        )
        if has_meta:
            continue
        if list(submod.parameters(recurse=False)):
            if dtype:
                submod.to(device=device, dtype=dtype)
            else:
                submod.to(device=device)


def _quantize(model, config, args):
    """Quantize layer-by-layer on CUDA, keeping the model on CPU.

    Only submodules without meta buffers are moved to CUDA. The quantized
    model stays on CPU — torch.export traces the graph without executing
    ops, so CUDA is not needed. Peak GPU memory is ~1 bf16 layer at a time.
    """
    from executorch.extension.llm.export.quantize import quantize_model_

    # Quantize MoE expert weights (packed INT4 for fused_moe kernel)
    if args.qlinear:
        _quantize_experts_int4(model, config, args.qlinear_group_size, use_hqq=args.hqq)

    # Untie lm_head/embedding so they can be quantized independently:
    # embedding uses index lookup (8w), lm_head uses matmul (4w).
    if model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())

    # Quantize transformer layers (skip meta buffers when moving to CUDA)
    for i, layer in enumerate(model.layers):
        _to_device_skip_meta(layer, device="cuda", dtype=torch.bfloat16)
        if args.qlinear:
            quantize_model_(
                layer,
                qlinear_config=args.qlinear,
                qlinear_group_size=args.qlinear_group_size,
                qlinear_packing_format=(
                    "tile_packed_to_4d" if args.qlinear == "4w" else None
                ),
            )
        _to_device_skip_meta(layer, device="cpu")
        torch.cuda.empty_cache()
        print(f"  Quantized layer {i + 1}/{config.num_hidden_layers}", end="\r")
    print()

    # Quantize lm_head (needs CUDA for tinygemm packing)
    if args.qlinear:
        print("Quantizing lm_head...")
        model.lm_head.to(device="cuda", dtype=torch.bfloat16)
        wrapper = nn.ModuleDict({"lm_head": model.lm_head})
        quantize_model_(
            wrapper,
            qlinear_config=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qlinear_packing_format=(
                "tile_packed_to_4d" if args.qlinear == "4w" else None
            ),
        )
        model.lm_head = wrapper.lm_head
        model.lm_head.to(device="cpu")
        torch.cuda.empty_cache()

    # Quantize embedding (doesn't need CUDA)
    if args.qembedding:
        print(f"Quantizing embeddings ({args.qembedding})...")
        model.embed_tokens.to(dtype=torch.bfloat16)
        quantize_model_(model, qembedding_config=args.qembedding)

    # Cast remaining unquantized modules
    model.norm.to(dtype=torch.bfloat16)

    if args.qlinear:
        print(f"Quantized linear layers ({args.qlinear})")


def _materialize_buffers(model, config):
    """Materialize meta-device buffers before torch.export.

    Replaces meta buffers with real tensors on CPU, recomputes RoPE
    inv_freq and causal masks.
    """
    # State buffers (KV cache, conv/recurrent state) are bf16 to match
    # compute dtype. Masks stay bool, inv_freq stays float32.
    for fqn, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            dtype = torch.bfloat16 if buf.dtype != torch.bool else torch.bool
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            parent.register_buffer(
                parts[-1],
                torch.zeros(buf.shape, dtype=dtype, device="cpu"),
            )

    # Recompute RoPE inv_freq (zero-fill above is wrong for these)
    for layer in model.layers:
        if hasattr(layer.attn, "rotary_emb"):
            rope = layer.attn.rotary_emb
            inv_freq = 1.0 / (
                config.rope_theta
                ** (
                    torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32)
                    / rope.rotary_dim
                )
            )
            rope.inv_freq = inv_freq

    # Recompute cache_positions for full attention layers
    for layer in model.layers:
        if hasattr(layer.attn, "cache_positions"):
            layer.attn.cache_positions = torch.arange(
                config.max_seq_len, dtype=torch.long
            )


def _apply_turboquant(model, config):
    """Replace KV caches in full-attention layers with TurboQuantKVCache.

    Runs after _materialize_buffers so the new TQ4 buffers are created
    with correct dtypes and not affected by any blanket cast in _quantize.
    """
    from executorch.extension.llm.modules.turboquant import TurboQuantKVCache

    count = 0
    for layer in model.layers:
        if layer.layer_type != "full_attention":
            continue
        old_cache = layer.attn.kv_cache
        _, n_heads, max_seq_len, head_dim = old_cache.k_cache.shape
        layer.attn.kv_cache = TurboQuantKVCache(
            n_heads,
            head_dim,
            max_seq_len,
        )
        layer.attn.turboquant = True
        count += 1

    print(f"Replaced {count} KV caches with TurboQuantKVCache (TQ4)")


# ---------------------------------------------------------------------------
# Export + lower
# ---------------------------------------------------------------------------


def export_and_lower(model, config, args):
    """Export model to .pte via torch.export + CUDA backend."""
    import torch._inductor.config as inductor_config

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    # Coordinate descent recompiles each kernel trying config perturbations,
    # adding minutes with negligible runtime benefit for this model's shapes.
    inductor_config.coordinate_descent_tuning = False
    # The wrapper.cpp is pure kernel launch orchestration — no heavy compute.
    # -O0 compiles ~8x faster than -O1 with no measurable runtime impact.
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"

    # Dynamic shapes for forward method
    example_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    example_input_pos = torch.tensor([0, 1], dtype=torch.long)
    seq_dim = Dim("seq_len", min=1, max=config.max_seq_len - 1)
    dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

    print("Exporting forward method with torch.export...")
    with torch.no_grad():
        exported_forward = export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print("Forward export successful!")

    # Export sample method by temporarily swapping model.forward
    print("Exporting sample method with torch.export...")
    original_forward = model.forward
    model.forward = model.sample
    example_logits = torch.zeros(1, 2, config.vocab_size, dtype=torch.bfloat16)
    example_temperature = torch.tensor([0.8], dtype=torch.float32)
    sample_dynamic_shapes = ({1: seq_dim}, None)
    with torch.no_grad():
        exported_sample = export(
            model,
            (example_logits, example_temperature),
            dynamic_shapes=sample_dynamic_shapes,
            strict=True,
        )
    model.forward = original_forward
    print("Sample export successful!")

    # Lower with CUDA backend (multi-method)
    print("Lowering to ExecuTorch with CUDA...")
    forward_compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    sample_compile_specs = [CudaBackend.generate_method_name_compile_spec("sample")]

    metadata = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }
    et_prog = to_edge_transform_and_lower(
        {"forward": exported_forward, "sample": exported_sample},
        partitioner={
            "forward": [CudaPartitioner(forward_compile_specs)],
            "sample": [CudaPartitioner(sample_compile_specs)],
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
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            enable_non_cpu_memory_planning=True,
            skip_h2d_for_method_inputs=True,
            skip_d2h_for_method_outputs=True,
        ),
    )

    # Save .pte
    os.makedirs(args.output_dir, exist_ok=True)
    pte_path = os.path.join(args.output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    # Save .ptd tensor data (CUDA backend)
    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(args.output_dir)
        print(f"Saved tensor data to {args.output_dir}/")

    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3.5 MoE to ExecuTorch (CUDA)"
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="HuggingFace model directory (not needed with --prequantized)",
    )
    parser.add_argument(
        "--output-dir", default="./qwen35_moe_exports", help="Output directory"
    )
    parser.add_argument("--max-seq-len", type=int, default=4096, help="KV cache length")
    parser.add_argument(
        "--qlinear",
        default=None,
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Quantize linear layers.",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=32,
        help="Group size for linear quantization.",
    )
    parser.add_argument(
        "--qembedding", default=None, choices=["8w"], help="Quantize embedding layers."
    )
    parser.add_argument(
        "--hqq",
        action="store_true",
        help="Use HQQ scale-only optimization for expert quantization (slower, better accuracy).",
    )
    parser.add_argument(
        "--prequantized",
        default=None,
        help="Path to prequantized directory (from quantize_and_save.py) "
        "containing model.safetensors and config.json. "
        "Skips quantization; --model-dir is not needed.",
    )
    parser.add_argument(
        "--turboquant",
        action="store_true",
        help="Enable TurboQuant TQ4 KV cache compression (3.8x cache savings).",
    )
    args = parser.parse_args()

    if not args.prequantized and not args.model_dir:
        parser.error("--model-dir is required unless --prequantized is provided.")

    if args.hqq and not args.qlinear:
        parser.error("--hqq requires --qlinear")

    # Register FLA Triton kernel
    import executorch.backends.cuda.triton.kernels  # noqa: F401

    model, config = load_and_quantize(args)
    _materialize_buffers(model, config)

    if args.turboquant:
        _apply_turboquant(model, config)

    export_and_lower(model, config, args)


if __name__ == "__main__":
    main()
