"""Quantize Qwen 3.5 MoE and save as a self-contained safetensors checkpoint.

Runs quantization once and saves the result so export.py can skip
re-quantizing via --prequantized. The output directory contains everything
needed to load the model — no reference to the original HF checkpoint required.

Output:
  output_dir/
    model.safetensors       # quantized weights (with reconstruction metadata in header)
    config.json             # model architecture config
    tokenizer.json          # tokenizer (for runtime)
    tokenizer_config.json
    merges.txt
    vocab.json

Usage:
  python quantize_and_save.py --model-dir /path/to/Qwen3.5-MoE-A3B --qlinear 4w
  python quantize_and_save.py --model-dir /path/to/model --qlinear 4w --hqq
  python quantize_and_save.py --model-dir /path/to/model --sensitive --hqq
"""

import argparse
import json
import os
import shutil

import torch

from executorch.examples.models.qwen3_5_moe.export import _quantize, _quantize_sensitive
from executorch.examples.models.qwen3_5_moe.model import Qwen35MoE
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Safetensors roundtrip for quantized models
#
# Tensor subclasses (Int4TilePackedTo4dTensor, IntxUnpackedToInt8Tensor) can't
# be stored directly in safetensors. We flatten them into plain inner tensors
# with .__<name> suffixes and store reconstruction metadata (class name,
# block_size, shape, dtypes) in the safetensors header under "quantization".
# ---------------------------------------------------------------------------

# Registry of tensor subclass types we know how to reconstruct.
_SUBCLASS_REGISTRY = {}


def _register_subclass(cls):
    _SUBCLASS_REGISTRY[cls.__qualname__] = cls


def _init_subclass_registry():
    """Lazily populate the registry on first use."""
    if _SUBCLASS_REGISTRY:
        return
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
        Int4TilePackedTo4dTensor,
    )

    _register_subclass(Int4TilePackedTo4dTensor)
    _register_subclass(IntxUnpackedToInt8Tensor)


def save_quantized_model(model, safetensors_path):
    """Save a quantized model to safetensors with subclass reconstruction metadata.

    Iterates named_parameters and named_buffers directly (no state_dict copy)
    to avoid doubling peak memory. Tensor subclasses are flattened into plain
    inner tensors with .__<name> suffixes.
    """
    tensors = {}
    subclass_meta = {}

    seen = set()
    for key, val in list(model.named_parameters()) + list(model.named_buffers()):
        if key in seen:
            continue
        seen.add(key)

        if val.device.type == "meta":
            continue

        if hasattr(val, "__tensor_flatten__"):
            inner_names, attrs = val.__tensor_flatten__()
            meta = {"_type": type(val).__qualname__}
            for attr_name, attr_val in attrs.items():
                if isinstance(attr_val, torch.Size):
                    meta[attr_name] = list(attr_val)
                elif isinstance(attr_val, torch.dtype):
                    meta[attr_name] = str(attr_val)
                else:
                    meta[attr_name] = attr_val
            subclass_meta[key] = meta

            for name in inner_names:
                inner_tensor = getattr(val, name)
                tensors[f"{key}.__{name}"] = inner_tensor.contiguous()
        else:
            tensors[key] = val.data.contiguous()

    header_metadata = {}
    if subclass_meta:
        header_metadata["quantization"] = json.dumps(subclass_meta)

    save_file(tensors, safetensors_path, metadata=header_metadata)
    return len(tensors)


def load_quantized_state_dict(safetensors_path):
    """Load a quantized state dict from safetensors, reconstructing tensor subclasses.

    Returns a state dict with plain tensors and reconstructed tensor subclasses
    ready for model.load_state_dict(state_dict, strict=False, assign=True).
    """
    from safetensors import safe_open

    _init_subclass_registry()

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        header_metadata = f.metadata()
        flat_tensors = {key: f.get_tensor(key) for key in f.keys()}

    quantization_meta = json.loads(header_metadata.get("quantization", "{}"))

    state_dict = {}
    reconstructed_keys = set()
    for key, meta in quantization_meta.items():
        cls = _SUBCLASS_REGISTRY[meta["_type"]]

        # Collect inner tensors
        tensor_data = {}
        prefix = f"{key}.__"
        for flat_key in list(flat_tensors.keys()):
            if flat_key.startswith(prefix):
                inner_name = flat_key[len(prefix) :]
                tensor_data[inner_name] = flat_tensors[flat_key]
                reconstructed_keys.add(flat_key)

        # Restore Python types from JSON (lists stay as lists, not tuples,
        # because Int4TilePackedTo4dTensor expects block_size as a list)
        attrs = {}
        for attr_name, attr_val in meta.items():
            if attr_name == "_type":
                continue
            elif attr_name == "shape":
                attrs[attr_name] = torch.Size(attr_val)
            elif isinstance(attr_val, str) and attr_val.startswith("torch."):
                attrs[attr_name] = getattr(torch, attr_val.split(".")[-1])
            else:
                attrs[attr_name] = attr_val

        # outer_size and outer_stride are unused by TorchAOBaseTensor.__tensor_unflatten__
        state_dict[key] = cls.__tensor_unflatten__(tensor_data, attrs, None, None)

    # Add plain tensors
    for key, tensor in flat_tensors.items():
        if key not in reconstructed_keys:
            state_dict[key] = tensor

    return state_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3.5 MoE and save as safetensors"
    )
    parser.add_argument(
        "--model-dir", required=True, help="HuggingFace model directory"
    )
    parser.add_argument(
        "--output",
        default="qwen35_moe_quantized",
        help="Output directory (default: qwen35_moe_quantized/)",
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
        help="Use HQQ scale-only optimization for expert quantization.",
    )
    parser.add_argument(
        "--sensitive",
        action="store_true",
        help="Use sensitivity-aware mixed precision quantization. "
        "Recommended for models without quantization-aware training.",
    )
    args = parser.parse_args()

    if not args.qlinear and not args.qembedding and not args.sensitive:
        parser.error(
            "At least one of --qlinear, --qembedding, or --sensitive is required."
        )

    if args.sensitive and (args.qlinear or args.qembedding):
        parser.error(
            "--sensitive manages its own precision; "
            "do not combine with --qlinear or --qembedding"
        )

    if args.hqq and not args.qlinear and not args.sensitive:
        parser.error("--hqq requires --qlinear or --sensitive")

    # Load model
    print("Loading model...")
    model, config = Qwen35MoE.from_hf_checkpoint(
        args.model_dir, max_seq_len=args.max_seq_len
    )
    model.eval()
    print(
        f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
        f"{config.num_experts} experts top-{config.num_experts_per_tok}"
    )

    # Quantize (includes expert INT4 + linear + embedding quantization)
    if args.sensitive:
        _quantize_sensitive(model, config, args)
    else:
        _quantize(model, config, args)

    # Save bundle
    os.makedirs(args.output, exist_ok=True)

    safetensors_path = os.path.join(args.output, "model.safetensors")
    print("Saving quantized weights...")
    n_tensors = save_quantized_model(model, safetensors_path)

    # Copy config and tokenizer from source
    for filename in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "LICENSE",
    ]:
        src = os.path.join(args.model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, filename))

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"Saved {n_tensors} tensors ({size_mb:.1f} MB) to {args.output}/")
    print(f"Done. Use with: python export.py --prequantized {args.output}")


if __name__ == "__main__":
    main()
