# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager inference on Gemma 4 31B-IT (CUDA + torch.compile).

Input paths:
  --prequantized <dir>   Load a quantized checkpoint (from quantize_and_save.py).
                         This is the path that supports the image (vision) flow.
  --gguf <file>          Load a GGUF file (e.g., Q4_K_M from the community).
                         Text-only (community GGUFs carry no vision tensors);
                         in-band GGUF vision is added in the g4-vision-llama-cpp
                         branch.
  --bf16 <dir>           Load the bf16 HF safetensors checkpoint via from_hf_checkpoint.

Gemma 4 31B-IT is instruction-tuned and requires chat-template formatting.
The ``--prompt`` is automatically wrapped with the Gemma 4 chat template
(``<|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>``; BOS is prepended separately).
Pass ``--raw-prompt`` to skip template wrapping (e.g., for pre-formatted input).

When ``--image-path`` is supplied the runner mirrors the C++ runner in
``main.cpp``: it patchifies the image, runs the vision tower (built as a
``Gemma4_31BVisionTower`` wrapper around ``model.vision_tower`` /
``model.embed_vision``), runs ``embed_text`` on the chat-template token
sequence, splices the vision embeddings into the rows where the
``<image>`` placeholder lives, and then prefills on the spliced embeds via
``model.forward(inputs_embeds, input_pos, temperature)``. Decode then
proceeds one token at a time through ``model.decode_forward``.

Usage:
    python inference.py \\
        --prequantized ./gemma4_31b_int4 \\
        --prompt "Write a short joke about saving RAM." \\
        --max-new-tokens 128 \\
        --temperature 0.8

    python inference.py \\
        --prequantized ./gemma4_31b_int4 \\
        --image-path ./some_image.png \\
        --prompt "Describe this image."

    python inference.py \\
        --gguf ./gemma-4-31B-it-Q4_K_M.gguf \\
        --tokenizer-path ./tokenizer.json \\
        --prompt "Hello"
"""

import argparse
import os
import time

import torch

from executorch.examples.models.gemma4.chat_template import (
    BOS_ID,
    build_vision_input_ids,
    IMAGE_TOKEN_ID,
)
from executorch.examples.models.gemma4_31b.export import load_prequantized_model
from executorch.examples.models.gemma4_31b.model import (
    Gemma4_31B,
    materialize_runtime_buffers,
)


def _move_to_cuda(model, config) -> None:
    """Move the prequantized model to CUDA and materialize runtime buffers there.

    Parameters are moved individually (not via ``model.cuda()``) to preserve
    ``Int4TilePackedTo4dTensor`` subclass identity. Non-meta buffers (e.g.
    ``layer_scalar``) are moved to CUDA. Meta-device buffers (KV cache, RoPE,
    constants) are materialized directly on CUDA via
    ``materialize_runtime_buffers``.
    """
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            # All checkpoints (prequant / GGUF / bf16) produce a fully
            # materialized text + vision model, so this branch is unreachable
            # in normal use. Surface it loudly if it ever trips.
            raise AssertionError(
                f"_move_to_cuda: parameter {name!r} is still on meta — "
                "checkpoint loader did not populate it."
            )
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


def apply_chat_template(prompt: str) -> str:
    """Wrap a user prompt in the Gemma 4 IT chat template.

    Does not include BOS — ``generate()`` prepends it at the token-ID level.
    """
    return (
        "<|turn>user\n"
        + prompt
        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    eos_token_ids=None,
    bos_token_id: int = 2,
) -> str:
    """Autoregressive generation. Prefill is one-token-at-a-time so a single
    compiled graph handles every step; the exported PTE uses a separate
    multi-token prefill method, but for eager+compile a uniform decode-shape
    forward is simpler and benefits from CUDA-graph friendly shapes.

    ``tokenizers.Tokenizer.from_file`` does not auto-prepend BOS — and Gemma 4
    is unusable without it (the model's logits collapse to a single
    high-frequency vocab token if the very first input isn't BOS). We prepend
    explicitly here; pass ``bos_token_id=None`` to disable.
    """
    if eos_token_ids is None:
        eos_token_ids = set()

    input_ids = tokenizer.encode(prompt).ids
    if bos_token_id is not None and (not input_ids or input_ids[0] != bos_token_id):
        input_ids = [bos_token_id] + input_ids

    temp_val = max(temperature, 1e-6)  # avoid div-by-zero in the on-device sampler
    temp_tensor = torch.tensor([temp_val], dtype=torch.float32, device="cuda")

    # The 4-method export contract changed `model.forward` to take pre-computed
    # embeddings (used by the unified prefill). For the per-token text-only
    # eager loop we use `decode_forward(tokens, pos, temperature)` instead,
    # which takes token inputs and internally runs `embed_text` -> `_run_blocks`.
    # `decode_forward` is NOT one of the methods that `torch.compile` wraps
    # (only `forward` is); we go through `_orig_mod` so attribute access lands
    # on the original module rather than the compile wrapper's proxy path.
    underlying = getattr(model, "_orig_mod", model)

    sampled = None
    with torch.no_grad():
        # Prefill, one token at a time.
        for i, tok_id in enumerate(input_ids):
            tok = torch.tensor([[tok_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([i], dtype=torch.long, device="cuda")
            sampled = underlying.decode_forward(tok, pos, temp_tensor)

        # First generated token from the last prefill step.
        next_id = int(sampled.item())
        generated = [next_id]

        # Decode loop.
        seq_len = len(input_ids)
        for i in range(max_new_tokens - 1):
            tok = torch.tensor([[next_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([seq_len + i], dtype=torch.long, device="cuda")
            sampled = underlying.decode_forward(tok, pos, temp_tensor)
            next_id = int(sampled.item())
            generated.append(next_id)
            if next_id in eos_token_ids:
                break

    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------


def _build_vision_encoder(model, config):
    """Build a ``Gemma4_31BVisionTower`` wrapper that reuses the model's already-
    loaded ``vision_tower`` and ``embed_vision`` submodules.

    Mirrors ``export.py::_build_vision_encoder_wrapper``: construct the wrapper
    on the meta device (so its freshly-built children take no real allocation),
    then swap in the loaded modules so parameter identity is preserved.
    """
    from executorch.examples.models.gemma4_31b.vision_tower import Gemma4_31BVisionTower

    # When ``model`` has been wrapped by ``torch.compile`` we still want the
    # raw underlying modules — torch.compile proxies attribute access to
    # ``_orig_mod``, so ``model.vision_tower`` already gives us the originals.
    underlying = getattr(model, "_orig_mod", model)

    with torch.device("meta"):
        wrapper = Gemma4_31BVisionTower(config.vision_config, config.hidden_size)
    wrapper.vision_tower = underlying.vision_tower
    wrapper.embed_vision = underlying.embed_vision
    wrapper.eval()
    return wrapper


def generate_with_image(
    model,
    vision_encoder,
    tokenizer,
    prompt: str,
    image_path: str,
    max_vision_soft_tokens: int = 280,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    eos_token_ids=None,
    bos_token_id: int = BOS_ID,
) -> str:
    """Image+text generation. Mirrors the C++ runner flow in main.cpp:

    1. Patchify image -> (pixel_values, pixel_position_ids).
    2. vision_encoder(pixels, position_ids) -> (image_embeds, pooler_mask).
    3. Build chat-template input_ids with ``num_soft_tokens`` image
       placeholders.
    4. embed_text(input_ids) -> text_embeds.
    5. Splice image_embeds into text_embeds at ``<image>`` rows.
    6. Single-shot prefill via model.forward(spliced, input_pos, temp).
    7. Decode loop via model.decode_forward(token, input_pos, temp).
    """
    from executorch.examples.models.gemma4.image_utils import preprocess_image

    if eos_token_ids is None:
        eos_token_ids = set()

    # 1. Patchify.
    pixel_values, pixel_position_ids, num_soft_tokens = preprocess_image(
        image_path, max_soft_tokens=max_vision_soft_tokens
    )
    pixel_values = pixel_values.to("cuda")
    pixel_position_ids = pixel_position_ids.to("cuda")
    print(
        f"Image: patchified to {pixel_values.shape[1]} patches; "
        f"{num_soft_tokens} soft tokens (max={max_vision_soft_tokens})."
    )

    underlying = getattr(model, "_orig_mod", model)

    temp_val = max(temperature, 1e-6)
    temp_tensor = torch.tensor([temp_val], dtype=torch.float32, device="cuda")

    with torch.no_grad():
        # 2. Vision tower.
        image_embeds, pooler_mask = vision_encoder(pixel_values, pixel_position_ids)
        # image_embeds: [1, output_length, hidden_size] bf16
        # pooler_mask:  [1, output_length] bool, True = valid soft token

        # 3. Token sequence.
        input_ids = build_vision_input_ids(
            tokenizer, prompt, num_soft_tokens, bos_id=bos_token_id
        )
        T = len(input_ids)
        tokens = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        print(f"Prompt tokens (image+text): {T}")

        # 4. embed_text.
        text_embeds = underlying.embed_text(tokens)  # [1, T, hidden] bf16

        # 5. Splice image rows into text_embeds at IMAGE_TOKEN_ID positions,
        # skipping any image-embed rows whose pooler_mask is False (padded
        # soft tokens).
        inputs_embeds = text_embeds.clone()
        valid_mask_row = pooler_mask[0]  # [output_length]
        n_image_rows = int(image_embeds.shape[1])
        image_idx = 0
        spliced = 0
        for i, tok_id in enumerate(input_ids):
            if tok_id != IMAGE_TOKEN_ID:
                continue
            # Advance to next valid soft-token row.
            while image_idx < n_image_rows and not bool(valid_mask_row[image_idx]):
                image_idx += 1
            if image_idx >= n_image_rows:
                raise RuntimeError(
                    f"Ran out of valid vision soft tokens at text position {i} "
                    f"(used {spliced}, needed {num_soft_tokens})."
                )
            inputs_embeds[0, i] = image_embeds[0, image_idx]
            image_idx += 1
            spliced += 1
        if spliced != num_soft_tokens:
            raise RuntimeError(
                f"Spliced {spliced} image rows but expected {num_soft_tokens}."
            )

        # 6. Single-shot prefill on spliced embeddings. We bypass the
        # torch.compile wrapper here: calling ``forward`` on the underlying
        # module executes uncompiled, but for prefill (one call) the cost
        # is negligible and avoids re-compiling for the variable T.
        input_pos = torch.arange(T, dtype=torch.long, device="cuda")
        sampled = underlying.forward(inputs_embeds, input_pos, temp_tensor)
        next_id = int(sampled.item())
        generated = [next_id]
        if next_id in eos_token_ids:
            return tokenizer.decode(generated)

        # 7. Decode loop. We use ``decode_forward`` (token-input single-step)
        # which the runner also uses for decode steps. Like prefill, this
        # bypasses the compile wrapper — fine for label generation.
        for i in range(max_new_tokens - 1):
            tok = torch.tensor([[next_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([T + i], dtype=torch.long, device="cuda")
            sampled = underlying.decode_forward(tok, pos, temp_tensor)
            next_id = int(sampled.item())
            generated.append(next_id)
            if next_id in eos_token_ids:
                break

    return tokenizer.decode(generated)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eager inference on Gemma 4 31B-IT.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--prequantized",
        default=None,
        help="Path to a quantized checkpoint directory.",
    )
    src.add_argument(
        "--gguf",
        default=None,
        help="Path to a GGUF file (e.g., gemma-4-31B-it-Q4_K_M.gguf).",
    )
    src.add_argument(
        "--bf16",
        default=None,
        help="Path to a bf16 hf directory (e.g., gemma-4-31B).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to tokenizer.json (required with --gguf, optional with --prequantized).",
    )
    parser.add_argument("--prompt", default="Hello", help="Input prompt.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = near-greedy).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="KV cache length to allocate for this run.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Skip chat-template wrapping (use if the prompt is already formatted).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (slower, but easier to debug).",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda"],
        help="Target backend.",
    )
    parser.add_argument(
        "--image-path",
        default="",
        help=(
            "Optional: path to an image file (JPEG/PNG). When set, the runner "
            "uses the multimodal flow: vision_tower + embed_text + spliced "
            "prefill (mirrors examples/models/gemma4_31b/main.cpp)."
        ),
    )
    parser.add_argument(
        "--max-vision-soft-tokens",
        type=int,
        default=280,
        help=(
            "Maximum number of vision soft tokens emitted by the vision "
            "tower. Must be one of {70,140,280,560,1120}. Default 280 matches "
            "the Gemma 4 vision tower default."
        ),
    )
    return parser


def _resolve_tokenizer_path(args, parser: argparse.ArgumentParser) -> str:
    if args.tokenizer_path:
        return args.tokenizer_path
    if args.prequantized:
        return os.path.join(args.prequantized, "tokenizer.json")
    if args.bf16:
        return os.path.join(args.bf16, "tokenizer.json")
    parser.error("--tokenizer-path is required with --gguf.")


def _load_model_from_args(args, parser: argparse.ArgumentParser):
    if args.gguf:
        from executorch.examples.models.gemma4_31b.gguf_loader import load_gguf_model

        return load_gguf_model(
            args.gguf,
            max_seq_len=args.max_seq_len,
            backend=args.backend,
        )
    if args.bf16:
        return Gemma4_31B.from_hf_checkpoint(args.bf16, max_seq_len=args.max_seq_len)

    print(f"Loading prequantized model from {args.prequantized}...")
    return load_prequantized_model(
        args.prequantized, max_seq_len=args.max_seq_len, backend=args.backend
    )


def _prepare_model_for_inference(model, config, args, parser: argparse.ArgumentParser):
    _move_to_cuda(model, config)
    model.eval()

    import executorch.backends.cuda.int4_dispatch  # noqa: F401

    # Build the vision encoder BEFORE wrapping the model with torch.compile —
    # the wrapper steals references to model.vision_tower / model.embed_vision,
    # and we want those references to stay valid no matter what we do with
    # ``model`` afterwards. (Building it after compile also works because
    # torch.compile proxies attribute access to _orig_mod, but doing it first
    # is clearer.)
    vision_encoder = None
    if args.image_path:
        if config.vision_config is None:
            parser.error(
                "Loaded model has no vision_config; cannot run with --image-path."
            )
        vision_encoder = _build_vision_encoder(model, config)

    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    return model, vision_encoder


def _run_generation(model, vision_encoder, tokenizer, args, prompt_str: str) -> str:
    # Gemma 4 EOS tokens (from generation_config.json: ids 1, 50, 106).
    eos_token_ids = {1, 50, 106}

    if args.image_path:
        return generate_with_image(
            model,
            vision_encoder,
            tokenizer,
            args.prompt,
            args.image_path,
            max_vision_soft_tokens=args.max_vision_soft_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_token_ids=eos_token_ids,
        )

    return generate(
        model,
        tokenizer,
        prompt_str,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eos_token_ids=eos_token_ids,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the cuda backend.")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(_resolve_tokenizer_path(args, parser))
    prompt_str = args.prompt if args.raw_prompt else apply_chat_template(args.prompt)

    model, config = _load_model_from_args(args, parser)
    model, vision_encoder = _prepare_model_for_inference(model, config, args, parser)

    print(f"\nPrompt: {args.prompt}")
    if args.image_path:
        print(f"Image:  {args.image_path}")
    print("-" * 40)

    t0 = time.perf_counter()
    output = _run_generation(model, vision_encoder, tokenizer, args, prompt_str)
    elapsed = time.perf_counter() - t0

    print(output)
    print("-" * 40)
    print(f"Generated in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
