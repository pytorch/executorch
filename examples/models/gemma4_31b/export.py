# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export Gemma 4 31B-IT to ExecuTorch (.pte + .ptd).

Two methods are exported and lowered together so they share KV-cache buffers:
  - "decode":  T=1, static shape, returns the next sampled token.
  - "prefill": T>=2, dynamic shape, returns the next sampled token.

Three input paths:
  --prequantized <dir>      Load a quantized checkpoint (from quantize_and_save.py)
                            and pack for the target backend. No re-quantization.
  --gguf <file>             Load a GGUF file (e.g., Q4_K_M from the community).
  --model-dir <hf>          Load bf16 checkpoint, quantize, pack, and export
                            in one shot.

Backends:
  --backend cuda            (default) CUDA via tinygemm INT4 + CudaPartitioner.
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


# ---------------------------------------------------------------------------
# Load paths


def load_prequantized_model(
    prequantized_dir: str,
    max_seq_len: int = 4096,
    backend: str = "cuda",
) -> tuple[Gemma4_31B, Gemma4_31BConfig]:
    """Load a quantized checkpoint and pack for the target backend."""
    config = Gemma4_31BConfig.from_hf_config(
        os.path.join(prequantized_dir, "config.json")
    )
    config.max_seq_len = max_seq_len

    print("Building model on meta device...")
    with torch.device("meta"):
        model = Gemma4_31B(config)

    safetensors_path = os.path.join(prequantized_dir, "model.safetensors")
    print(f"Loading quantized checkpoint from {safetensors_path}...")
    _pack_for_backend(model, safetensors_path, backend)
    model.eval()

    print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
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
    state_dict = quantize_model(model, recipe)

    print(f"Packing for {backend}...")
    with torch.device("meta"):
        model = Gemma4_31B(config)
    pack_model(model, state_dict, packers=_get_packers(backend))
    model.eval()

    print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
    return model, config


# ---------------------------------------------------------------------------
# Backend dispatch helpers


def _get_packers(backend: str) -> dict:
    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.quant import DEFAULT_CUDA_PACKERS

        return DEFAULT_CUDA_PACKERS
    raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'cuda'.")


def _pack_for_backend(model: nn.Module, path: str, backend: str) -> None:
    if backend == "cuda":
        from executorch.examples.models.gemma4_31b.quant import load_and_pack_for_cuda

        load_and_pack_for_cuda(path, model)
    else:
        raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'cuda'.")


# ---------------------------------------------------------------------------
# Export + lower


def export_and_lower(
    model: Gemma4_31B,
    config: Gemma4_31BConfig,
    output_dir: str,
    backend: str = "cuda",
) -> None:
    """Export and lower the model to ExecuTorch for the given backend."""
    if backend == "cuda":
        _export_cuda(model, config, output_dir)
    else:
        raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'cuda'.")


def _export_cuda(model: Gemma4_31B, config: Gemma4_31BConfig, output_dir: str) -> None:
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

    inductor_config.coordinate_descent_tuning = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = "O0"

    from executorch.backends.cuda.transforms.int4_linear_dispatch import (
        use_tinygemm_linears,
    )

    materialize_runtime_buffers(model, dtype=torch.bfloat16)

    # Prefill first (T>=2): default IntxUnpacked dispatch does dequant+cuBLAS,
    # which is optimal for large M (compute-bound).
    max_prefill = min(config.max_seq_len - 1, config.sliding_window * 2)
    seq_dim = Dim("seq_len", min=2, max=max_prefill)
    print(f"Exporting prefill (T in [2, {max_prefill}])...")
    with torch.no_grad():
        prefill_ep = export(
            model,
            (
                torch.zeros((1, max_prefill), dtype=torch.long),
                torch.arange(max_prefill, dtype=torch.long),
                torch.tensor([1.0], dtype=torch.float32),
            ),
            dynamic_shapes=({1: seq_dim}, {0: seq_dim}, None),
            strict=True,
        )

    # Decode second (T=1): convert to tinygemm, optimal for M=1 (bandwidth-bound).
    print("Converting INT4 linears to tinygemm for decode...")
    use_tinygemm_linears(model)
    print("Exporting decode (T=1)...")
    with torch.no_grad():
        decode_ep = export(
            model,
            (
                torch.tensor([[0]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.float32),
            ),
            strict=True,
        )

    print("Lowering to ExecuTorch with CUDA backend...")
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
    print(f"  {os.path.getsize(pte_path) / 1024**2:.1f} MB")

    if et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)
        print(f"  Saved tensor data (.ptd) to {output_dir}/")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    from executorch.examples.models.gemma4_31b.quantize_and_save import _RECIPES

    parser = argparse.ArgumentParser(description="Export Gemma 4 31B-IT to ExecuTorch.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model-dir",
        default=None,
        help="HuggingFace model dir. Triggers load + quantize + export.",
    )
    src.add_argument(
        "--prequantized",
        default=None,
        help="Path to a quantized checkpoint directory. Skips quantization.",
    )
    src.add_argument(
        "--gguf",
        default=None,
        help="Path to a GGUF file (e.g., gemma-4-31B-it-Q4_K_M.gguf).",
    )
    parser.add_argument(
        "--output-dir",
        default="./gemma4_31b_exports",
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
        choices=list(_RECIPES),
        help="Quantization recipe (only with --model-dir).",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda"],
        help="Target backend for export.",
    )
    args = parser.parse_args()

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

    export_and_lower(model, config, args.output_dir, backend=args.backend)


if __name__ == "__main__":
    main()
