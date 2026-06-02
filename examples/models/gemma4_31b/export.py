# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Gemma 4 31B-IT to ExecuTorch (.pte + .ptd).

CUDA backend — embeddings-based 4-method vision contract
========================================================
On CUDA, four methods are exported and lowered together so they share the
underlying nn.Parameter / mutable-buffer storage:

  - "embed_text"     (tokens [1,T] i64) -> (embeds [1,T,5376] bf16)
                     Pure embedding lookup + sqrt(hidden_size) scale.
  - "vision_encoder" (pixel_values [1,P,768] f32, pixel_position_ids [1,P,2] i64)
                     -> (image_embeds [1,N,5376] bf16, mask [1,N] bool)
                     Backed by Gemma4_31BVisionTower (vision_tower + embed_vision).
                     P = 9 * N (avg-pool requirement); default max N = 280.
  - "prefill"        (inputs_embeds [1,T,5376] bf16, input_pos [T] i64,
                     temperature [1] f32) -> sampled [1,1] f32
                     Unified entry for text-only AND image+text. The runner
                     builds inputs_embeds from embed_text + vision_encoder splice.
  - "decode"         (tokens [1,1] i64, input_pos [1] i64, temperature [1] f32)
                     -> sampled [1,1] f32
                     Token-input single-step (model.decode_forward).

MLX backend — text-only (this branch)
=====================================
MLX still exports ``main``'s single token-input method (dynamic seq_len, host
sampling) realized via a temporary fake-prefill wrapper, and drops the vision
head before lowering. MLX vision support is added in the g4-vision-mlx branch.

Three input paths:
  --prequantized <dir>      Load a quantized checkpoint (from quantize_and_save.py)
                            and pack for the target backend. No re-quantization.
                            Includes vision keys (used by the CUDA vision methods).
  --gguf <file>             Load a GGUF file (e.g., Q4_K_M from the community).
  --model-dir <hf>          Load bf16 checkpoint, quantize, pack, and export
                            in one shot.

Backends:
  --backend cuda            (default) CUDA via tinygemm INT4 + CudaPartitioner.
                            4-method vision contract.
  --backend mlx             Apple Silicon via MLXPartitioner (single text-only
                            method this branch; host-side sampling).
"""

import argparse
import os

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.model import (
    Gemma4_31B,
    Gemma4_31BConfig,
    materialize_runtime_buffers,
)
from executorch.examples.models.gemma4_31b.pack_vision import (
    pack_vision_patch_embedder,
    quantize_vision_position_table,
)
from executorch.examples.models.gemma4_31b.vision_tower import Gemma4VisionPatchEmbedder


# ---------------------------------------------------------------------------
# Load paths


def _checkpoint_has_int8_vision_pe(safetensors_path: str) -> bool:
    """Return True when the checkpoint stores the vision PE table as int8 buffers."""
    from safetensors import safe_open

    pet_int8_key = "vision_tower.patch_embedder._pet_int8"
    pet_scale_key = "vision_tower.patch_embedder._pet_scale"
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())

    has_int8 = pet_int8_key in keys
    has_scale = pet_scale_key in keys
    if has_int8 != has_scale:
        missing = pet_scale_key if has_int8 else pet_int8_key
        raise RuntimeError(
            "Incomplete quantized vision position-embedding table in checkpoint: "
            f"missing {missing!r}."
        )
    return has_int8


def load_prequantized_model(
    prequantized_dir: str,
    max_seq_len: int = 4096,
    backend: str = "cuda",
) -> tuple[Gemma4_31B, Gemma4_31BConfig]:
    """Load a quantized checkpoint and pack for the target backend.

    Vision is mandatory: the checkpoint MUST contain vision keys. The model
    is built with vision_tower / embed_vision attached and the safetensors
    must populate them; otherwise loading raises.

    Args:
        prequantized_dir: dir produced by quantize_and_save.py.
        max_seq_len: KV-cache size.
        backend: target backend ("cuda" only today).
    """
    config = Gemma4_31BConfig.from_hf_config(
        os.path.join(prequantized_dir, "config.json")
    )
    config.max_seq_len = max_seq_len

    safetensors_path = os.path.join(prequantized_dir, "model.safetensors")

    print("Building model on meta device...")
    with torch.device("meta"):
        model = Gemma4_31B(config)

    # Gemma4-specific packers install the vision PE int8 dispatch when they
    # see ``_pet_int8`` / ``_pet_scale`` keys, so no explicit install call is
    # needed here.
    print(f"Loading quantized checkpoint from {safetensors_path}...")
    _pack_for_backend(model, safetensors_path, backend)
    model.eval()

    print(
        f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}"
        " (+ vision tower)"
    )
    return model, config


def load_and_quantize(
    model_dir: str,
    recipe_name: str,
    max_seq_len: int = 4096,
    backend: str = "cuda",
) -> tuple[Gemma4_31B, Gemma4_31BConfig]:
    """Load bf16 checkpoint, quantize, pack — one shot."""
    from executorch.examples.models.gemma4_31b.quant import pack_model, quantize_model
    from executorch.examples.models.gemma4_31b.quantize_and_save import _RECIPES

    recipe = _RECIPES[recipe_name]

    print("Loading checkpoint (lazy, shard-by-shard)...")
    model, config = Gemma4_31B.from_hf_checkpoint(model_dir, max_seq_len=max_seq_len)

    if model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr():
        print("Untying embed_tokens / lm_head...")
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())

    print(f"Quantizing with recipe '{recipe_name}'...")
    quantize_vision_position_table(model.vision_tower)
    state_dict = quantize_model(model, recipe, verbose=True)

    print(f"Packing for {backend}...")
    with torch.device("meta"):
        model = Gemma4_31B(config)
    pack_model(model, state_dict, packers=_get_packers(backend))
    model.eval()

    print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
    return model, config


# ---------------------------------------------------------------------------
# Backend dispatch helpers


_SUPPORTED_BACKENDS = ("cuda", "mlx")


def _get_packers(backend: str) -> dict:
    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.quant import DEFAULT_CUDA_PACKERS

        return {
            **DEFAULT_CUDA_PACKERS,
            Gemma4VisionPatchEmbedder: pack_vision_patch_embedder,
        }
    if backend == "mlx":
        from executorch.examples.models.gemma4_31b.quant import DEFAULT_MLX_PACKERS

        return {
            **DEFAULT_MLX_PACKERS,
            Gemma4VisionPatchEmbedder: pack_vision_patch_embedder,
        }
    raise ValueError(
        f"Unsupported backend: {backend!r}. Supported: {_SUPPORTED_BACKENDS}."
    )


def _pack_for_backend(model: nn.Module, path: str, backend: str) -> None:
    packers = _get_packers(backend)
    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.quant import load_and_pack_for_cuda

        load_and_pack_for_cuda(path, model, packers=packers)
    elif backend == "mlx":
        from executorch.examples.models.gemma4_31b.quant import load_and_pack_for_mlx

        load_and_pack_for_mlx(path, model, packers=packers)
    else:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Supported: {_SUPPORTED_BACKENDS}."
        )


# ---------------------------------------------------------------------------
# Vision encoder wrapper construction
#
# We build a Gemma4_31BVisionTower wrapper on the meta device (no real
# allocation) and then SUBSTITUTE its freshly-constructed submodules with the
# already-loaded modules from `model`. This guarantees parameter identity with
# the rest of the model so the cross-method weight dedupe pass sees a single
# underlying storage.


def _build_vision_encoder_wrapper(
    model: Gemma4_31B, config: Gemma4_31BConfig
) -> nn.Module:
    from executorch.examples.models.gemma4_31b.vision_tower import Gemma4_31BVisionTower

    with torch.device("meta"):
        wrapper = Gemma4_31BVisionTower(config.vision_config, config.hidden_size)
    # Replace the freshly-built (meta) submodules with the already-loaded ones.
    # nn.Module.__setattr__ deregisters the old child and registers the new one.
    wrapper.vision_tower = model.vision_tower
    wrapper.embed_vision = model.embed_vision
    wrapper.eval()
    return wrapper


# ---------------------------------------------------------------------------
# Multimodal entry-point — instance-attr swap
#
# The 4 export contract methods all live on the SAME `Gemma4_31B` instance.
# To export each one with `torch.export.export(model, ...)` (which only takes
# an `nn.Module`, not a bound method), we temporarily monkey-patch the bound
# method onto `model.forward` for the duration of the export call. Critical:
#
#   * The model's class identity does NOT change — all 4 ExportedPrograms
#     export from the SAME `Gemma4_31B` instance with identical buffer FQNs
#     (`layers.X.self_attn.kv_cache.k_cache` etc.). This is what lets
#     `to_executorch(share_mutable_buffers=True)` unify the prefill / decode
#     KV-cache buffers under ONE underlying tensor at runtime — without
#     which prefill writes the cache and decode reads from a fresh-zero
#     cache (the bug runner_dev hit).
#   * `nn.Module.__call__` invokes `self.forward(...)` via normal attribute
#     lookup, so an instance-attribute `forward` shadows the class method
#     during the export call. We restore (`del model.forward`) afterwards.
#
# This replaces the previous `model.__class__ = _Gemma4_31BFor*` swap, which
# created different subclasses per method and fragmented mutable-buffer
# identity at lowering time.


class _BoundMethodForward:
    """Context manager: temporarily set ``model.forward`` to a bound method."""

    def __init__(self, model: nn.Module, bound_method) -> None:
        self._model = model
        self._bound = bound_method

    def __enter__(self):
        self._model.forward = self._bound  # instance-attr; shadows the class method
        return self._model

    def __exit__(self, *exc):
        # `del` restores the class method via the descriptor protocol.
        try:
            del self._model.forward
        except AttributeError:
            pass
        return False


def _drop_vision_head(model: nn.Module) -> None:
    """Detach the multimodal head before a text-only (MLX) lowering.

    Gemma 4 31B is multimodal-by-default, so ``vision_tower`` / ``embed_vision``
    are always constructed and loaded. The MLX path is text-only this branch, so
    we delete them here BEFORE ``materialize_runtime_buffers`` (which only
    rebuilds the vision RoPE table when ``vision_tower`` is still attached).
    """
    for name in ("vision_tower", "embed_vision"):
        if hasattr(model, name):
            delattr(model, name)


# ---------------------------------------------------------------------------
# Export + lower


def export_and_lower(
    model: Gemma4_31B,
    config: Gemma4_31BConfig,
    output_dir: str,
    backend: str = "cuda",
    use_turboquant: bool = False,
    *,
    max_vision_soft_tokens: int = 280,
) -> None:
    """Export and lower the model to ExecuTorch for the given backend."""
    if backend == "cuda":
        if use_turboquant:
            raise ValueError(
                "--turboquant is only supported with --backend mlx "
                "(the CUDA path here uses a different TurboQuant integration; "
                "see examples/models/qwen3_5_moe/export.py)."
            )
        _export_cuda(
            model,
            config,
            output_dir,
            max_vision_soft_tokens=max_vision_soft_tokens,
        )
    elif backend == "mlx":
        _export_mlx(model, config, output_dir, use_turboquant=use_turboquant)
    else:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Supported: {_SUPPORTED_BACKENDS}."
        )


def _export_cuda(
    model: Gemma4_31B,
    config: Gemma4_31BConfig,
    output_dir: str,
    *,
    max_vision_soft_tokens: int,
) -> None:
    import gc

    import torch._inductor.config as inductor_config

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"

    # Register Int4Tensor dispatch → executorch_cuda::int4_plain_mm shim
    import executorch.backends.cuda.int4_dispatch  # noqa: F401

    materialize_runtime_buffers(model, dtype=torch.bfloat16)

    # The 4-method contract REQUIRES vision — vision_tower and embed_vision
    # must be present on the model.
    assert config.vision_config is not None, (
        "4-method export requires vision_config. Did you load via the "
        "text-only --model-dir or --gguf path?"
    )

    # Reset CUDA peak memory counter so EXPORT_GPU_PEAK_MEMORY_MB reflects
    # the export itself (not setup or quantization upstream).
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Int4Tensor weights are used directly — no format conversion. F.linear
    # dispatches to executorch_cuda::int4_plain_mm (CUDA shim). All methods
    # share the same nibble-packed weights.

    max_prefill = min(config.max_seq_len - 1, config.sliding_window * 2)
    hidden_size = config.hidden_size

    # ---------- prefill (inputs_embeds, dynamic seq_len) ----------
    # Uses `Gemma4_31B.forward` which already takes inputs_embeds.
    seq_dim = Dim("seq_len", min=5, max=max_prefill)
    print(f"Exporting prefill (T in [5, {max_prefill}], inputs_embeds)...")
    with torch.no_grad():
        prefill_ep = export(
            model,
            (
                torch.zeros((1, max_prefill, hidden_size), dtype=torch.bfloat16),
                torch.arange(max_prefill, dtype=torch.long),
                torch.tensor([1.0], dtype=torch.float32),
            ),
            dynamic_shapes=({1: seq_dim}, {0: seq_dim}, None),
            strict=True,
        )

    # ---------- decode (tokens, T=1, static) ----------
    # Bound-method swap (NOT __class__ swap) so the model instance's class —
    # and therefore its mutable-buffer FQN identity — is preserved across
    # all 4 ExportedPrograms. This is what lets share_mutable_buffers unify
    # the prefill / decode KV-cache buffers under one runtime tensor.
    print("Exporting decode (T=1, tokens input)...")
    with _BoundMethodForward(model, model.decode_forward), torch.no_grad():
        decode_ep = export(
            model,
            (
                torch.tensor([[0]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.float32),
            ),
            strict=True,
        )

    # ---------- embed_text ----------
    print(f"Exporting embed_text (T in [1, {max_prefill}])...")
    embed_text_seq_dim = Dim("embed_text_seq_len", min=1, max=max_prefill)
    with _BoundMethodForward(model, model.embed_text), torch.no_grad():
        embed_text_ep = export(
            model,
            (torch.zeros((1, max_prefill), dtype=torch.long),),
            dynamic_shapes=({1: embed_text_seq_dim},),
            strict=True,
        )

    # ---------- vision_encoder ----------
    pks = config.vision_config.pooling_kernel_size
    max_patches = (pks * pks) * max_vision_soft_tokens
    patch_dim = config.vision_config.patch_dim  # e.g. 3*16*16 = 768
    print(
        f"Exporting vision_encoder "
        f"(P in [{pks*pks}, {max_patches}], soft tokens up to "
        f"{max_vision_soft_tokens})..."
    )
    ve_wrapper = _build_vision_encoder_wrapper(model, config)
    num_groups_dim = Dim("vision_num_groups", min=1, max=max_vision_soft_tokens)
    num_patches_dim = (pks * pks) * num_groups_dim
    with torch.no_grad():
        vision_encoder_ep = export(
            ve_wrapper,
            (
                # Per locked contract: pixel_values is float32; the
                # patch_embedder internally casts to weight.dtype (bf16).
                torch.zeros((1, max_patches, patch_dim), dtype=torch.float32),
                torch.zeros((1, max_patches, 2), dtype=torch.long),
            ),
            dynamic_shapes=(
                {1: num_patches_dim},
                {1: num_patches_dim},
            ),
            strict=True,
        )
    del ve_wrapper

    programs: dict[str, "torch.export.ExportedProgram"] = {
        "embed_text": embed_text_ep,
        "vision_encoder": vision_encoder_ep,
        "prefill": prefill_ep,
        "decode": decode_ep,
    }

    del model
    gc.collect()

    # Per-method partitioners. Same CudaPartitioner / low_memory_mode for all.
    def _cuda_partitioner_for(method_name: str) -> CudaPartitioner:
        return CudaPartitioner(
            [
                CudaBackend.generate_method_name_compile_spec(method_name),
                CompileSpec("low_memory_mode", b"ON"),
            ]
        )

    partitioner_map: dict[str, list[CudaPartitioner]] = {
        name: [_cuda_partitioner_for(name)] for name in programs
    }
    transform_passes: dict[str, list] = {name: [] for name in programs}

    constant_methods: dict[str, object] = {
        "get_max_seq_len": config.max_seq_len,
        "get_vocab_size": config.vocab_size,
        "get_n_layers": config.num_hidden_layers,
        "get_max_prefill_chunk": max_prefill,
        "get_max_vision_soft_tokens": int(max_vision_soft_tokens),
        "use_kv_cache": True,
        "use_sdpa_with_kv_cache": False,
        "enable_dynamic_shape": True,
    }

    print(
        f"Lowering {len(programs)} methods to ExecuTorch with CUDA backend: "
        f"{', '.join(programs.keys())}..."
    )
    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner_map,
        transform_passes=transform_passes,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    del programs, prefill_ep, decode_ep, embed_text_ep, vision_encoder_ep
    gc.collect()

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

    del et_prog
    gc.collect()

    # GPU peak memory marker — emitted on its own line for CI grep.
    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"EXPORT_GPU_PEAK_MEMORY_MB: {peak_mb:.1f}")

    os.makedirs(output_dir, exist_ok=True)
    pte_path = os.path.join(output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    print(f"  {os.path.getsize(pte_path) / 1024**2:.1f} MB")

    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)
        print(f"  Saved tensor data (.ptd) to {output_dir}/")
    print("Done.")


def _export_mlx(
    model: Gemma4_31B,
    config: Gemma4_31BConfig,
    output_dir: str,
    use_turboquant: bool = False,
) -> None:
    """Export to .pte via torch.export + MLX backend (text-only this branch).

    Exports a single token-input method with dynamic sequence length; MLX
    samples on the host so there is no temperature input. The vision head is
    dropped first and ``mlx_source_transformations`` installs the token-input
    ``forward`` (``main``'s contract). MLX vision is added in the g4-vision-mlx
    branch.

    When ``use_turboquant=True``, full-attention layers swap to
    ``MLXTurboQuantKVCache`` for ~3.8x KV cache memory savings. Sliding
    layers are unaffected (already use ``RingBufferKVCache``).
    """
    import gc

    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes

    from executorch.examples.models.gemma4_31b.mlx_source_transformations import (
        mlx_source_transformations,
    )
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import Dim, export

    # Text-only contract: drop the vision head before transforming / exporting.
    _drop_vision_head(model)

    mlx_source_transformations(
        model, dtype=torch.bfloat16, use_turboquant=use_turboquant
    )

    materialize_runtime_buffers(model, dtype=torch.bfloat16)

    max_prefill = 256
    seq_dim = Dim("seq_len", min=1, max=max_prefill)

    print(f"Exporting (T in [1, {max_prefill}])...")
    with torch.no_grad():
        exported = export(
            model,
            (
                torch.tensor([[0, 1]], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            ),
            dynamic_shapes=({1: seq_dim}, {0: seq_dim}),
            strict=True,
        )

    del model
    gc.collect()

    print("Lowering to ExecuTorch with MLX backend...")
    et_prog = to_edge_transform_and_lower(
        exported,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods={
            "get_max_seq_len": config.max_seq_len,
            "get_vocab_size": config.vocab_size,
            "get_n_layers": config.num_hidden_layers,
            "get_max_prefill_chunk": max_prefill,
            "use_kv_cache": True,
            "use_sdpa_with_kv_cache": False,
            "enable_dynamic_shape": True,
        },
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

    os.makedirs(output_dir, exist_ok=True)
    pte_path = os.path.join(output_dir, "model.pte")
    print(f"Saving to {pte_path}...")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)
    print(f"  {os.path.getsize(pte_path) / 1024**2:.1f} MB")

    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)
        print(f"  Saved tensor data (.ptd) to {output_dir}/")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Gemma 4 31B-IT to ExecuTorch (4-method contract)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model-dir",
        default=None,
        help="HuggingFace model dir. Triggers load + quantize + export. "
        "Text-only path — incompatible with the 4-method contract.",
    )
    src.add_argument(
        "--prequantized",
        default=None,
        help="Path to a quantized checkpoint directory (with vision keys). "
        "Skips quantization. This is the supported path for the 4-method "
        "contract.",
    )
    src.add_argument(
        "--gguf",
        default=None,
        help="Path to a GGUF file (text-only — incompatible with the "
        "4-method contract).",
    )
    parser.add_argument(
        "--output-dir",
        default="./gemma4_31b_vision_exports",
        help="Output directory for model.pte / model.ptd.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="KV cache size.",
    )
    parser.add_argument(
        "--quant-recipe",
        default="default",
        choices=["default", "sensitive"],
        help="Quantization recipe (only with --model-dir).",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=list(_SUPPORTED_BACKENDS),
        help="Target backend for export.",
    )
    parser.add_argument(
        "--turboquant",
        action="store_true",
        help="Use TurboQuant TQ4 KV cache compression (MLX backend only). "
        "~3.8× cache memory savings; applies only to full-attention "
        "(non-sliding) layers — sliding layers keep RingBufferKVCache.",
    )
    parser.add_argument(
        "--max-vision-soft-tokens",
        type=int,
        default=280,
        help="Maximum vision soft tokens per image (= dynamic upper bound on "
        "vision_encoder output length). The corresponding upper bound on "
        "input patches is pooling_kernel_size**2 * this value (default: "
        "9 * 280 = 2520 patches).",
    )
    args = parser.parse_args()

    if args.turboquant and args.backend != "mlx":
        parser.error("--turboquant requires --backend mlx.")
    if args.backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the cuda backend.")

    if args.prequantized:
        model, config = load_prequantized_model(
            args.prequantized,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
        )
    elif args.gguf:
        from executorch.examples.models.gemma4_31b.gguf_loader import load_gguf_model

        model, config = load_gguf_model(
            args.gguf, max_seq_len=args.max_seq_len, backend=args.backend
        )
    else:
        model, config = load_and_quantize(
            args.model_dir,
            args.quant_recipe,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
        )

    if args.gguf and args.backend == "mlx":
        os.environ["ET_MLX_ALLOW_NON_FUSED_QUANTIZED_OPS"] = "1"
    try:
        export_and_lower(
            model,
            config,
            args.output_dir,
            backend=args.backend,
            use_turboquant=args.turboquant,
            max_vision_soft_tokens=args.max_vision_soft_tokens,
        )
    finally:
        os.environ.pop("ET_MLX_ALLOW_NON_FUSED_QUANTIZED_OPS", None)


if __name__ == "__main__":
    main()
