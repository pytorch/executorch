"""Quantize Gemma4-31B layer-by-layer and save as a safetensors bundle.

Mirrors the qwen3_5_moe recipe: keep the model on CPU, ship one decoder
layer at a time to CUDA for quantization, then move it back. Peak GPU usage
is ~1 bf16 layer (~1 GB for Gemma4-31B), avoiding the 78 GB bf16 footprint
that OOMs the lowering pass.

Output layout:
  output_dir/
    model.safetensors       # quantized weights + reconstruction metadata
    config.json
    tokenizer.json / .model  (whichever the source has)

Usage:
  python quantize_and_save.py --model-dir /path/to/gemma-4-31B \\
      --output /home/gasoonjia/models/gemma4_31B_int4_hqq
"""

import argparse
import os
import shutil

import torch
import torch.nn as nn

from executorch.examples.models.gemma4.export import load_full_model
from executorch.examples.models.qwen3_5_moe.quantize_and_save import (
    save_quantized_model,
)


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


def _quantize_dense(model, config, qlinear, group_size):
    """Per-layer quantize the 60 decoder layer Linears on CUDA, then move back to CPU.

    Skipped on purpose:
      - embed_tokens: Gemma4ScaledEmbedding is not nn.Embedding (quantize_model_
        wouldn't match it) and Gemma4 was trained with tied weights, so it
        also acts as the lm_head — leave both bf16 to preserve the tie.
      - lm_head: stays tied to embed_tokens (bf16). Untying + INT4-quantizing
        lm_head while embedding stays bf16 produces asymmetric reconstruction
        and degenerate outputs (verified: collapses to a single special token).

    For qlinear="4w" + tile_packed_to_4d the underlying Int4WeightOnlyConfig
    uses HQQ (int4_choose_qparams_algorithm="hqq") unconditionally.
    """
    from executorch.extension.llm.export.quantize import quantize_model_

    packing = "tile_packed_to_4d" if qlinear == "4w" else None

    for i, layer in enumerate(model.layers):
        _to_device_skip_meta(layer, device="cuda", dtype=torch.bfloat16)
        quantize_model_(
            layer,
            qlinear_config=qlinear,
            qlinear_group_size=group_size,
            qlinear_packing_format=packing,
        )
        _to_device_skip_meta(layer, device="cpu")
        torch.cuda.empty_cache()
        print(
            f"  Quantized layer {i + 1}/{config.num_hidden_layers}",
            end="\r",
            flush=True,
        )
    print()

    # embed_tokens is the same Parameter object as lm_head.weight (tied).
    model.embed_tokens.to(dtype=torch.bfloat16)
    model.norm.to(dtype=torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Gemma4 and save as safetensors"
    )
    parser.add_argument("--model-dir", required=True, help="HF Gemma4 directory")
    parser.add_argument(
        "--output",
        default="gemma4_31B_int4_hqq",
        help="Output directory for the prequant bundle",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=4096, help="KV cache length"
    )
    parser.add_argument(
        "--qlinear",
        default="4w",
        choices=["4w", "8w", "8da4w", "8da8w"],
        help="Linear quantization scheme (default: 4w)",
    )
    parser.add_argument(
        "--qlinear-group-size",
        type=int,
        default=128,
        help="Group size for linear quantization (default: 128)",
    )
    args = parser.parse_args()

    print(f"Loading Gemma4 from {args.model_dir} (CPU, bf16)...")
    # load_full_model casts to bf16 and loads on CPU via meta-device assign.
    model, config = load_full_model(args.model_dir, args.max_seq_len)
    print(
        f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
        f"vocab={config.vocab_size}"
    )

    # from_hf_checkpoint(assign=True) creates two distinct Parameter objects
    # sharing storage for tied embeddings; safetensors rejects that. Re-tie
    # so save_quantized_model only writes embed_tokens.weight (the tie is
    # restored after load by load_prequantized_model).
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    _quantize_dense(
        model,
        config,
        qlinear=args.qlinear,
        group_size=args.qlinear_group_size,
    )

    os.makedirs(args.output, exist_ok=True)
    safetensors_path = os.path.join(args.output, "model.safetensors")
    print("Saving quantized weights...")
    n_tensors = save_quantized_model(model, safetensors_path)

    for filename in [
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        src = os.path.join(args.model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, filename))

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"Saved {n_tensors} tensors ({size_mb:.1f} MB) to {args.output}/")


if __name__ == "__main__":
    main()
