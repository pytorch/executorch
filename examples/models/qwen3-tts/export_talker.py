"""Export Qwen3-TTS talker backbone to ExecuTorch.

The talker is architecturally identical to Qwen3 0.6B (same attention, MLP,
RMSNorm, QK-norm, RoPE) so we reuse the existing Llama/Qwen3 export
infrastructure directly.

This exports the main talker as a standard autoregressive LM with KV cache,
producing a .pte that supports prefill + per-token decode.

Usage:
    python export_talker.py \
        --checkpoint examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted/talker_main.pth \
        --params examples/models/qwen3-tts/config/talker_config.json \
        --output-dir examples/models/qwen3-tts/qwen3_tts_exports_talker \
        --backend xnnpack \
        --qlinear 8da4w
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.export import export

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qwen3-TTS talker to ExecuTorch."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Converted talker checkpoint (talker_main.pth).",
    )
    parser.add_argument(
        "--params", type=Path, required=True,
        help="Model params JSON (talker_config.json).",
    )
    parser.add_argument(
        "--backend", choices=["portable", "xnnpack"], default="xnnpack",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./qwen3_tts_exports_talker"),
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Max sequence length for KV cache allocation.",
    )
    parser.add_argument(
        "--qlinear", choices=["4w", "8w", "8da4w", "8da8w"], default=None,
    )
    parser.add_argument("--qlinear-group-size", type=int, default=32)
    parser.add_argument(
        "--output-name", type=str, default="talker.pte",
        help="Output .pte filename.",
    )
    parser.add_argument(
        "--no-embedding", action="store_true",
        help="Don't apply tok_embeddings (model takes hidden states). "
        "Used for code_predictor which has per-group embeddings.",
    )
    parser.add_argument(
        "--no-output", action="store_true",
        help="Don't apply output projection (model returns hidden states). "
        "Used for code_predictor which has per-group LM heads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from executorch.examples.models.llama.model_args import ModelArgs
    from executorch.examples.models.llama.llama_transformer import construct_transformer

    # Load config.
    with args.params.open("r") as f:
        params_dict = json.load(f)

    params_dict["use_kv_cache"] = True
    params_dict["max_seq_len"] = args.max_seq_len
    params_dict["max_context_len"] = args.max_seq_len
    params_dict["max_batch_size"] = 1
    params_dict["generate_full_logits"] = False
    if args.no_embedding:
        params_dict["apply_embedding"] = False
    if args.no_output:
        params_dict["apply_output"] = False

    model_args = ModelArgs(**params_dict)
    print(f"ModelArgs: dim={model_args.dim}, n_layers={model_args.n_layers}, "
          f"n_heads={model_args.n_heads}, n_kv_heads={model_args.n_kv_heads}, "
          f"vocab_size={model_args.vocab_size}, max_seq_len={model_args.max_seq_len}")

    # Build model.
    model = construct_transformer(model_args)
    model.eval()

    # Load weights.
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out KV cache buffers (expected to be missing).
        real_missing = [k for k in missing if "k_cache" not in k and "v_cache" not in k and "mask" not in k]
        if real_missing:
            print(f"WARNING: Missing keys: {real_missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    # Apply quantization.
    if args.qlinear is not None:
        from executorch.extension.llm.export.quantize import quantize_model_
        print(f"Applying {args.qlinear} quantization (group_size={args.qlinear_group_size})...")
        quantize_model_(
            model,
            qlinear_config=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
        )

    # Disable gradients on all parameters (required for in-place KV cache ops).
    for param in model.parameters():
        param.requires_grad_(False)
    for buf in model.buffers():
        buf.requires_grad_(False)

    # Export with KV cache: single-token decode mode.
    example_attn_options = {"input_pos": torch.tensor([0], dtype=torch.long)}

    if args.no_embedding:
        # Code predictor: takes hidden states [1, 1, dim] instead of token ids.
        example_h = torch.randn(1, 1, model_args.dim)
        example_args = (None, example_attn_options, example_h)
    else:
        # Main talker: takes token ids [1, 1].
        example_tokens = torch.tensor([[0]], dtype=torch.long)
        example_args = (example_tokens, example_attn_options)

    print("Exporting with torch.export...")
    with torch.no_grad():
        exported = export(
            model,
            example_args,
            strict=False,
        )

    # Lower to ExecuTorch.
    if args.backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )
        partitioner = [XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]
    else:
        partitioner = []

    constant_methods = {
        "max_seq_len": args.max_seq_len,
        "vocab_size": model_args.vocab_size,
        "dim": model_args.dim,
        "n_heads": model_args.n_heads,
        "n_kv_heads": model_args.n_kv_heads,
        "head_dim": model_args.head_dim,
        "n_layers": model_args.n_layers,
    }

    print("Lowering to ExecuTorch...")
    edge_prog = to_edge_transform_and_lower(
        {"forward": exported},
        partitioner={"forward": partitioner},
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods,
    )
    et_prog = edge_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=not args.no_output,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )

    model_path = args.output_dir / args.output_name
    with model_path.open("wb") as f:
        et_prog.write_to_file(f)
    print(f"Saved: {model_path}")

    manifest = {
        "model_type": "qwen3_tts_talker",
        "backend": args.backend,
        "qlinear": args.qlinear,
        "max_seq_len": args.max_seq_len,
        "model_args": params_dict,
        "constant_methods": constant_methods,
    }
    manifest_name = args.output_name.replace(".pte", "_manifest.json")
    manifest_path = args.output_dir / manifest_name
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
