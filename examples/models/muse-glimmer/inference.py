# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager inference on Muse Glimmer (CUDA + torch.compile).

Usage:
    python inference.py \
        --checkpoint-dir /path/to/consolidated \
        --tokenizer-path /path/to/hf_assets \
        --prompt "The meaning of life is" \
        --max-new-tokens 128
"""

import argparse
import time
from pathlib import Path

import torch
from executorch.examples.models.muse_glimmer.export.export_solo import (
    load_prequantized_model,
)
from executorch.examples.models.muse_glimmer.model.model import (
    materialize_runtime_buffers,
    MuseGlimmerModel,
)
from tokenizers import Tokenizer

BOS_TOKEN_ID = 200000
EOS_TOKEN_ID = 200001


def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load the Hugging Face tokenizer used by the runners."""
    path = Path(tokenizer_dir)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {path}")
    return Tokenizer.from_file(str(path))


def _move_to_cuda(model: MuseGlimmerModel) -> None:
    """Move weights to CUDA, preserving tensor subclass identity."""
    for name, p in model.named_parameters():
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        setattr(
            parent,
            parts[-1],
            torch.nn.Parameter(p.data.to("cuda"), requires_grad=False),
        )

    for fqn, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            parent.register_buffer(parts[-1], buf.to("cuda"), persistent=False)

    materialize_runtime_buffers(model, dtype=torch.bfloat16, device="cuda")


def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature > 0:
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()
    return logits.argmax().item()


def generate(
    model: MuseGlimmerModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    bos_token_id: int = BOS_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
) -> str:
    """Autoregressive generation with chunked multi-token prefill."""
    tokens = [bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
    max_chunk = model._sliding_window * 2
    print(f"Input: {len(tokens)} tokens")

    with torch.no_grad():
        # Chunked prefill: respects ring buffer size limit
        pos = 0
        while pos < len(tokens):
            chunk_len = min(len(tokens) - pos, max_chunk)
            chunk = tokens[pos : pos + chunk_len]
            input_ids = torch.tensor([chunk], dtype=torch.long, device="cuda")
            input_pos = torch.arange(
                pos, pos + chunk_len, dtype=torch.long, device="cuda"
            )
            logits = model(input_ids, input_pos)
            pos += chunk_len

        next_id = _sample(logits[0, -1].float(), temperature)
        generated = [next_id]

        # Decode loop: one token at a time
        seq_len = len(tokens)
        for step in range(max_new_tokens - 1):
            if next_id == eos_token_id:
                break

            tok = torch.tensor([[next_id]], dtype=torch.long, device="cuda")
            input_pos = torch.tensor([seq_len + step], dtype=torch.long, device="cuda")
            logits = model(tok, input_pos)
            next_id = _sample(logits[0, -1].float(), temperature)
            generated.append(next_id)

            if (step + 1) % 20 == 0:
                print(f"  Generated {step + 1} tokens...")

    return tokenizer.decode(generated, skip_special_tokens=False)


def _load_image_normalized(image_path: str, config) -> torch.Tensor:
    """Load an image, resize to the vision grid, normalize to [-1, 1].

    Returns a ``[3, H, W]`` float tensor where H, W are multiples of
    ``patch_size * downsample_factor`` (28), matching the eager preprocessing
    (mean/std 0.5).
    """
    from executorch.examples.models.muse_glimmer.vision.precompute import (
        compute_grid_size,
    )
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    target_h, target_w, _ = compute_grid_size(img.width, img.height, config)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    buf = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = buf.reshape(target_h, target_w, 3).permute(2, 0, 1).float()
    # Normalize [0,255] -> [-1,1] with mean/std 0.5.
    return (t / 255.0 - 0.5) / 0.5


def generate_with_image(
    model: MuseGlimmerModel,
    vision_model,
    pos_embed_table: torch.Tensor,
    tokenizer: Tokenizer,
    prompt: str,
    image_path: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    bos_token_id: int = BOS_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
) -> str:
    """Multimodal generation: encode the image, splice soft tokens, decode.

    Mirrors the C++ runner path: run the vision encoder on host-precomputed
    inputs, build ``[BOS] <|image_start|> <|patch|>*N <|image_end|> {prompt}``,
    embed the tokens, overwrite the patch rows with vision features, then run
    the unified ``prefill_from_embeds`` + per-token decode.
    """
    from executorch.examples.models.muse_glimmer.vision.precompute import (
        MuseGlimmerVisionConfig,
        precompute_vision_inputs,
    )

    PATCH_ID = tokenizer.token_to_id("<|patch|>")
    if PATCH_ID is None:
        raise ValueError("Tokenizer has no <|patch|> token")

    vcfg = MuseGlimmerVisionConfig()
    # Host precompute runs on CPU (grid_sample needs table + grid co-located);
    # move the resulting graph inputs to CUDA afterward.
    image = _load_image_normalized(image_path, vcfg)
    inputs = precompute_vision_inputs(image, pos_embed_table.cpu(), vcfg)
    args = tuple(
        a.to("cuda") if isinstance(a, torch.Tensor) else a for a in inputs.as_args()
    )
    with torch.no_grad():
        image_embeds = vision_model(*args)  # [1, N, hidden]
    n_soft = image_embeds.shape[1]
    print(f"Vision: {inputs.patches.shape[1]} patches -> {n_soft} soft tokens")

    text_tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    tokens = [bos_token_id] + [PATCH_ID] * n_soft + text_tokens
    max_chunk = model._sliding_window * 2

    with torch.no_grad():
        tok_t = torch.tensor([tokens], dtype=torch.long, device="cuda")
        embeds = model.embed_text(tok_t)  # [1, T, hidden]
        # Splice image rows at <|patch|> positions.
        patch_positions = [i for i, t in enumerate(tokens) if t == PATCH_ID]
        assert len(patch_positions) == n_soft
        for j, i in enumerate(patch_positions):
            embeds[0, i] = image_embeds[0, j].to(embeds.dtype)

        pos = 0
        logits = None
        while pos < len(tokens):
            chunk_len = min(len(tokens) - pos, max_chunk)
            chunk = embeds[:, pos : pos + chunk_len]
            input_pos = torch.arange(
                pos, pos + chunk_len, dtype=torch.long, device="cuda"
            )
            logits = model.prefill_from_embeds(chunk, input_pos)
            pos += chunk_len

        next_id = _sample(logits[0, -1].float(), temperature)
        generated = [next_id]

        seq_len = len(tokens)
        for step in range(max_new_tokens - 1):
            if next_id == eos_token_id:
                break
            tok = torch.tensor([[next_id]], dtype=torch.long, device="cuda")
            input_pos = torch.tensor([seq_len + step], dtype=torch.long, device="cuda")
            logits = model(tok, input_pos)
            next_id = _sample(logits[0, -1].float(), temperature)
            generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eager inference on Muse Glimmer.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to consolidated bf16 checkpoint.",
    )
    src.add_argument(
        "--prequantized",
        default=None,
        help="Path to a quantized checkpoint directory.",
    )
    src.add_argument(
        "--gguf",
        default=None,
        help="Path to a GGUF checkpoint (fused, quantized, packed for CUDA).",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Hugging Face asset directory or tokenizer.json path.",
    )
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-seq-len", type=int, default=16384, help="KV cache length."
    )
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile.")
    parser.add_argument(
        "--mmproj",
        default=None,
        help="Optional mmproj (clip/muse_glimmer) GGUF for the vision encoder. Enables "
        "image input via --image.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional path to an image file. Requires --mmproj.",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda"],
        help="Target backend.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.image and not args.mmproj:
        parser.error("--image requires --mmproj (the vision encoder weights).")

    if args.prequantized:
        print(f"Loading prequantized model from {args.prequantized}...")
        model, config = load_prequantized_model(
            args.prequantized,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
        )
    elif args.gguf:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_gguf_model,
        )

        print(f"Loading GGUF model from {args.gguf}...")
        model, config = load_gguf_model(
            args.gguf,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
        )
    else:
        print(f"Loading model from {args.checkpoint_dir}...")
        model, config = MuseGlimmerModel.from_checkpoint(
            args.checkpoint_dir, max_seq_len=args.max_seq_len
        )
    _move_to_cuda(model)
    model.eval()

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401

    vision_model = None
    pos_embed_table = None
    if args.mmproj:
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mmproj_vision_model,
        )

        print(f"Loading vision encoder from {args.mmproj}...")
        vision_model, pos_embed_table, _ = load_mmproj_vision_model(args.mmproj)
        _move_to_cuda(vision_model)
        vision_model.eval()

    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    tokenizer = load_tokenizer(args.tokenizer_path)

    print(f"\nPrompt: {args.prompt}")
    if args.image:
        print(f"Image: {args.image}")
    print("-" * 40)

    t0 = time.perf_counter()
    if args.image:
        output = generate_with_image(
            model,
            vision_model,
            pos_embed_table,
            tokenizer,
            args.prompt,
            args.image,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        output = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elapsed = time.perf_counter() - t0

    print(output)
    print("-" * 40)
    print(f"Generated in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
