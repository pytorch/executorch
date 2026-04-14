"""
Export Qwen 3.5 MoE to ExecuTorch .pte format.

Supports CUDA and MLX backends.

Usage:
  python export.py --model-id Qwen/Qwen3.5-35B-A3B
  python export.py --model-dir /path/to/Qwen3.5-MoE-A3B
  python export.py --model-dir /path/to/model --qlinear 4w
  python export.py --prequantized /path/to/quantized_bundle/
  python export.py --model-id Qwen/Qwen3.5-35B-A3B --backend mlx --qlinear 4w
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


def _prepare_and_quantize_mlx(model, config, args):
    """MLX: apply source transforms, quantize via torchao, pack experts."""
    from executorch.backends.mlx.llm.switch import pack_all_switch_linears
    from executorch.examples.models.qwen3_5_moe.mlx_source_transformations import (
        mlx_source_transformations,
    )

    model.to(dtype=torch.bfloat16)

    # Materialize meta-device buffers before source transforms
    for fqn, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            parent.register_buffer(
                parts[-1],
                torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"),
            )

    mlx_source_transformations(
        model,
        model_dtype=torch.bfloat16,
        config=config,
        sort_experts=True,
        fuse_gate_up=False,
    )
    if args.qlinear or args.qembedding:
        from executorch.extension.llm.export.quantize import quantize_model_

        quantize_model_(
            model,
            qlinear_config=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qembedding_config=args.qembedding,
            qembedding_group_size=getattr(args, "qembedding_group_size", None),
        )
    pack_all_switch_linears(model)


def _prepare_and_quantize_metal(model, config, args):
    """Metal: apply source transforms, quantize experts + non-expert layers."""
    import executorch.backends.apple.metal.ops.gated_delta_rule  # noqa: F401
    import executorch.backends.apple.metal.ops.gather_qmv  # noqa: F401
    from executorch.examples.models.qwen3_5_moe.metal_source_transformations import (
        metal_source_transformations,
        quantize_experts_metal,
    )

    # Quantize expert weights to Metal-compatible INT4 format
    if args.qlinear:
        quantize_experts_metal(model, config, args.qlinear_group_size)

    # Untie lm_head/embedding for independent quantization
    if model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())

    # Quantize non-expert layers with fpa4w (Metal-compatible, no CUDA needed).
    # Custom filter skips shared_expert_gate (N=1) which violates fpa4w's
    # N%4==0 constraint during prefill (M>1).
    if args.qlinear:
        import torchao.experimental.ops.mps  # noqa: F401
        from torchao.experimental.quant_api import UIntxWeightOnlyConfig
        from torchao.quantization.quant_api import quantize_

        fpa4w_config = UIntxWeightOnlyConfig(
            group_size=args.qlinear_group_size,
            bitwidth=4,
            uintx_choose_qparams_algorithm="hqq",
        )

        def _fpa4w_filter(mod, fqn):
            if not isinstance(mod, nn.Linear):
                return False
            n, k = mod.weight.shape
            if k % args.qlinear_group_size != 0:
                return False
            if n < 4:
                return False
            return True

        for i, layer in enumerate(model.layers):
            layer.to(dtype=torch.bfloat16)
            quantize_(layer, fpa4w_config, filter_fn=_fpa4w_filter)
            print(
                f"  Quantized layer {i + 1}/{config.num_hidden_layers} (fpa4w)",
                end="\r",
            )
        print()

        # Quantize lm_head
        print("Quantizing lm_head (fpa4w)...")
        from executorch.extension.llm.export.quantize import quantize_model_

        model.lm_head.to(dtype=torch.bfloat16)
        wrapper = nn.ModuleDict({"lm_head": model.lm_head})
        quantize_model_(
            wrapper,
            qlinear_config="fpa4w",
            qlinear_group_size=args.qlinear_group_size,
        )
        model.lm_head = wrapper.lm_head

    # Quantize embedding
    if args.qembedding:
        from executorch.extension.llm.export.quantize import quantize_model_

        print(f"Quantizing embeddings ({args.qembedding})...")
        model.embed_tokens.to(dtype=torch.bfloat16)
        quantize_model_(model, qembedding_config=args.qembedding)

    model.norm.to(dtype=torch.bfloat16)

    _materialize_buffers(model, config)
    metal_source_transformations(model, config=config)


def load_and_quantize(args):  # noqa: C901
    """Load model from checkpoint, optionally quantize.

    For CUDA: quantizes experts with packed INT4, then transformer layers on CUDA.
    For MLX: applies source transforms first, then quantizes via torchao, then packs.

    Returns (model, config) ready for export.
    """
    backend = getattr(args, "backend", "cuda")

    if not args.prequantized:
        if getattr(args, "tiny_test", False):
            # Build tiny model with random weights for CI testing.
            # Exercises the same architectural features as the real model:
            #   - GQA in full attention (n_heads=4, n_kv_heads=2 → 2:1 ratio)
            #   - GDN key/value head ratio (k_heads=2, v_heads=4 → 1:2 ratio)
            #   - Partial RoPE (25% of head_dim)
            #   - Mixed attention (full_attention_interval=2 → alternating layers)
            #   - Top-k MoE routing (top_k=2 from 8 experts)
            #   - Shared expert with gating
            #   - Fused gate+up expert weights [E, 2*inter, D]
            #   - Depthwise conv1d with state (kernel_dim=4)
            tiny_config = Qwen35MoEConfig(
                vocab_size=256,
                hidden_size=128,
                num_hidden_layers=4,  # 4 layers: 2 linear + 2 full attention
                num_attention_heads=4,  # GQA: 4 heads with 2 KV heads (2:1 ratio)
                num_kv_heads=2,
                head_dim=64,
                partial_rotary_factor=0.25,
                linear_num_key_heads=2,  # GDN: 2 key heads, 4 value heads (1:2 ratio)
                linear_num_value_heads=4,
                linear_key_head_dim=64,
                linear_value_head_dim=64,
                linear_conv_kernel_dim=4,
                num_experts=8,  # 8 experts, top-2 routing
                num_experts_per_tok=2,
                moe_intermediate_size=128,
                shared_expert_intermediate_size=128,
                full_attention_interval=2,  # alternating: linear, full, linear, full
                rms_norm_eps=1e-6,
                rope_theta=10_000.0,
                max_seq_len=64,
            )
            print("Building tiny model with random weights...")
            torch.manual_seed(42)
            model = Qwen35MoE(tiny_config)
            model.to(dtype=torch.bfloat16)
            for p in model.parameters():
                if p.device.type != "meta":
                    p.data.normal_(0, 0.02)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            config = tiny_config
            print(
                f"Tiny model: {config.num_hidden_layers} layers, "
                f"{config.num_experts} experts top-{config.num_experts_per_tok}, "
                f"layer_types={config.layer_types}"
            )
        else:
            print("Loading model...")
            model, config = Qwen35MoE.from_hf_checkpoint(
                args.model_dir, max_seq_len=args.max_seq_len
            )
            model.eval()
            print(
                f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
                f"{config.num_experts} experts top-{config.num_experts_per_tok}"
            )

    if backend == "mlx":
        if args.prequantized:
            raise ValueError(
                "MLX backend does not support custom prequantized weights. Use a prequantized torchao checkpoint instead."
            )
        _prepare_and_quantize_mlx(model, config, args)

    elif backend == "metal":
        if args.prequantized:
            return load_prequantized_model(args.prequantized, args.max_seq_len)
        _prepare_and_quantize_metal(model, config, args)

    elif backend == "cuda":
        if args.prequantized:
            return load_prequantized_model(args.prequantized, args.max_seq_len)

        # CUDA: quantize experts with packed INT4 for Triton kernel
        if args.qlinear or args.qembedding:
            _quantize(model, config, args)
        else:
            model.to(dtype=torch.bfloat16)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

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
        ".mask",
        ".inv_freq",
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
    inv_freq and causal masks. State buffers (KV cache, conv/recurrent
    state) are zero-initialized registered buffers that will be shared
    across methods via share_mutable_buffers.
    """
    # Masks stay bool, inv_freq stays float32.
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
    """Export model to .pte via torch.export + backend-specific lowering."""
    backend = getattr(args, "backend", "cuda")

    if backend == "mlx":
        _export_mlx(model, config, args)
    elif backend == "metal":
        _export_metal(model, config, args)
    else:
        _export_cuda(model, config, args)


def _export_mlx(model, config, args):
    """Export model to .pte via torch.export + MLX backend."""
    import gc

    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    example_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    example_input_pos = torch.tensor([0, 1], dtype=torch.long)
    seq_dim = Dim("seq_len", min=1, max=config.max_seq_len - 1)
    dynamic_shapes = ({1: seq_dim}, {0: seq_dim})

    print("Exporting with torch.export...")
    with torch.no_grad():
        exported = export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    print("Export successful!")

    del model
    gc.collect()

    print("Lowering to ExecuTorch with MLX backend...")
    metadata = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }
    et_prog = to_edge_transform_and_lower(
        exported,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )

    del exported
    gc.collect()

    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    del et_prog
    gc.collect()

    os.makedirs(args.output_dir, exist_ok=True)
    pte_path = os.path.join(args.output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"Saved {size_mb:.1f} MB")

    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(args.output_dir)
        print(f"Saved tensor data to {args.output_dir}/")

    print("Done!")


def _export_metal(model, config, args):
    """Export model to .pte via torch.export + Metal backend."""
    import torch._inductor.config as inductor_config

    from executorch.backends.apple.metal.metal_backend import MetalBackend
    from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"

    # --- Decode method (T=1, static shape) ---
    print("Exporting decode method...")
    decode_tokens = torch.tensor([[0]], dtype=torch.long)
    decode_pos = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        decode_ep = export(model, (decode_tokens, decode_pos), strict=True)
    print("Decode export successful!")

    # --- Prefill method (T>=2, dynamic shape) ---
    print("Exporting prefill method...")
    prefill_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    prefill_pos = torch.tensor([0, 1], dtype=torch.long)
    seq_dim = Dim("seq_len", min=2, max=config.max_seq_len - 1)
    prefill_dynamic_shapes = ({1: seq_dim}, {0: seq_dim})
    with torch.no_grad():
        prefill_ep = export(
            model,
            (prefill_tokens, prefill_pos),
            dynamic_shapes=prefill_dynamic_shapes,
            strict=True,
        )
    print("Prefill export successful!")

    # Lower with Metal backend
    print("Lowering to ExecuTorch with Metal...")
    metadata = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }
    et_prog = to_edge_transform_and_lower(
        {"decode": decode_ep, "prefill": prefill_ep},
        partitioner={
            "decode": [
                MetalPartitioner(
                    [MetalBackend.generate_method_name_compile_spec("decode")]
                )
            ],
            "prefill": [
                MetalPartitioner(
                    [MetalBackend.generate_method_name_compile_spec("prefill")]
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
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
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

    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(args.output_dir)
        print(f"Saved tensor data to {args.output_dir}/")

    print("Done!")


def _export_cuda(model, config, args):
    """Export model to .pte via torch.export + CUDA backend.

    Exports two methods:
      - "decode": decode path (T=1), uses native PyTorch recurrent FLA
        so AOTI can fuse with surrounding ops for maximum decode throughput.
      - "prefill": prefill path (T>=2), uses chunked FLA triton_op with
        dynamic sequence length.

    Both methods share mutable state buffers (KV cache, conv_state,
    recurrent_state) via share_mutable_buffers=True. The model uses
    registered buffers with in-place updates — no state in/out args.
    """
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

    # --- Decode method (T=1, static shape) ---
    print("Exporting decode method...")
    decode_tokens = torch.tensor([[0]], dtype=torch.long)
    decode_pos = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        decode_ep = export(
            model,
            (decode_tokens, decode_pos),
            strict=True,
        )
    print("Decode export successful!")

    # --- Prefill method (T>=2, dynamic shape) ---
    print("Exporting prefill method...")
    prefill_tokens = torch.tensor([[0, 1]], dtype=torch.long)
    prefill_pos = torch.tensor([0, 1], dtype=torch.long)
    seq_dim = Dim("seq_len", min=2, max=config.max_seq_len - 1)
    prefill_dynamic_shapes = (
        {1: seq_dim},  # tokens
        {0: seq_dim},  # input_pos
    )
    with torch.no_grad():
        prefill_ep = export(
            model,
            (prefill_tokens, prefill_pos),
            dynamic_shapes=prefill_dynamic_shapes,
            strict=True,
        )
    print("Prefill export successful!")

    # Lower with CUDA backend (per-method partitioners to avoid so_blob collision)
    print("Lowering to ExecuTorch with CUDA...")

    metadata = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }
    et_prog = to_edge_transform_and_lower(
        {"decode": decode_ep, "prefill": prefill_ep},
        partitioner={
            "decode": [
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("decode")]
                )
            ],
            "prefill": [
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("prefill")]
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


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Export Qwen3.5 MoE to ExecuTorch")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="HuggingFace model directory (not needed with --prequantized)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HuggingFace model-id",
    )
    parser.add_argument(
        "--output-dir", default="./qwen35_moe_exports", help="Output directory"
    )
    parser.add_argument("--max-seq-len", type=int, default=4096, help="KV cache length")
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "mlx", "metal"],
        help="Backend for export: cuda (default), mlx, or metal.",
    )
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
        "--qembedding-group-size",
        type=int,
        default=None,
        help="Group size for embedding quantization.",
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
    parser.add_argument(
        "--tiny-test",
        action="store_true",
        default=False,
        help="Build a tiny model with random weights for CI pipeline testing. "
        "No checkpoint download needed. Tests all architectural features "
        "(GQA, GDN head ratio, mixed attention, MoE routing) at small scale.",
    )
    args = parser.parse_args()

    if args.model_id:
        if args.model_dir is not None:
            raise ValueError("Cannot specify model_dir when model_id is provided.")
        from huggingface_hub import snapshot_download

        args.model_dir = snapshot_download(repo_id=args.model_id)

    if not args.prequantized and not args.model_dir and not args.tiny_test:
        parser.error(
            "--model-dir is required unless --prequantized or --tiny-test is provided."
        )

    if args.hqq and not args.qlinear:
        parser.error("--hqq requires --qlinear")

    if args.backend == "cuda":
        # Register FLA Triton kernel (CUDA only)
        import executorch.backends.cuda.triton.kernels  # noqa: F401

    if args.backend == "mlx":
        if args.prequantized:
            parser.error("--prequantized is not supported with --backend mlx")
        if args.turboquant:
            parser.error("--turboquant is not supported with --backend mlx")

    if args.backend == "metal":
        if args.turboquant:
            parser.error("--turboquant is not supported with --backend metal")

    model, config = load_and_quantize(args)

    if args.backend == "cuda":
        _materialize_buffers(model, config)
        if args.turboquant:
            _apply_turboquant(model, config)

    export_and_lower(model, config, args)


if __name__ == "__main__":
    main()
