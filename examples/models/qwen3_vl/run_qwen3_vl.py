#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL Multimodal Python Binding Example

Runs Qwen3-VL multimodal inference using ExecuTorch Python bindings for the
text decoder and PyTorch eager for the vision encoder.  The vision encoder
uses Conv3d (for 3D patch embedding), which the ExecuTorch portable runtime
does not yet support, so we run it through the original HF model.

The text decoder (token_embedding + text_decoder) runs entirely via the
exported .pte file.

Example usage:
    python run_qwen3_vl.py \
        --model_path Qwen3-VL-2B-Instruct-xnnpack-ori/model.pte \
        --image_path /path/to/image.jpg \
        --prompt "What is in this image?"
"""

import argparse
import sys
import time

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

try:
    import executorch.kernels.quantized  # noqa: F401
except Exception:
    pass

import torch as _torch
from executorch.extension.pybindings.portable_lib import _get_operator_names

if not any("quantized_decomposed" in op for op in _get_operator_names()):
    from pathlib import Path
    import site

    for sp in site.getsitepackages():
        candidates = list(
            Path(sp).glob("executorch/kernels/quantized/*quantized_ops_aot_lib*")
        )
        if candidates:
            _torch.ops.load_library(candidates[0])
            break

try:
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
except Exception:
    pass

from executorch.extension.pybindings.portable_lib import _load_for_executorch


def find_image_token_span(input_ids, image_token_id):
    """Return (start, end_exclusive) of the contiguous image-token run."""
    ids = input_ids.tolist()
    start = None
    end = None
    for i, t in enumerate(ids):
        if t == image_token_id:
            if start is None:
                start = i
            end = i + 1
    return start, end


def prefill_text(module, input_ids_1d, pos):
    """Embed text tokens via PTE and run a decoder prefill step."""
    token_ids = input_ids_1d.unsqueeze(0).to(torch.long)
    embeds = module.run_method("token_embedding", [token_ids])[0]
    seq_len = embeds.shape[1]
    cache_pos = torch.arange(pos, pos + seq_len, dtype=torch.long)
    logits = module.run_method("text_decoder", [embeds, cache_pos])[0]
    return logits, pos + seq_len


def prefill_image_embeds(module, image_embeds, pos):
    """Prefill the decoder with pre-computed image embeddings."""
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(0)
    seq_len = image_embeds.shape[1]
    cache_pos = torch.arange(pos, pos + seq_len, dtype=torch.long)
    logits = module.run_method("text_decoder", [image_embeds, cache_pos])[0]
    return logits, pos + seq_len


def decode_one(module, token_id, pos):
    """Run one autoregressive decode step."""
    token_ids = torch.tensor([[token_id]], dtype=torch.long)
    embeds = module.run_method("token_embedding", [token_ids])[0]
    cache_pos = torch.tensor([pos], dtype=torch.long)
    logits = module.run_method("text_decoder", [embeds, cache_pos])[0]
    return logits, pos + 1


def run_vision_encoder_eager(model_id, pixel_values, image_grid_thw):
    """Run the vision encoder using the HF model in eager mode.

    This is needed because the Qwen3-VL vision encoder uses Conv3d which
    the ExecuTorch portable runtime does not yet support.
    """
    print(f"Loading HF model vision encoder from: {model_id}")
    t0 = time.perf_counter()
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
    )
    print(f"HF model loaded in {time.perf_counter() - t0:.2f}s")

    with torch.no_grad():
        result = hf_model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        # get_image_features returns (tuple_of_embeds, deepstack_list).
        # The primary embeddings are result[0][0].
        if isinstance(result, (tuple, list)) and isinstance(result[0], (tuple, list)):
            image_embeds = result[0][0]
        elif isinstance(result, (tuple, list)):
            image_embeds = result[0]
        else:
            image_embeds = result

    del hf_model
    return image_embeds


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL multimodal inference with ExecuTorch",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="What is in this image?")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    args = parser.parse_args()

    # --- Load PTE ---
    print(f"Loading PTE from: {args.model_path}")
    t0 = time.perf_counter()
    module = _load_for_executorch(args.model_path)
    print(f"PTE loaded in {time.perf_counter() - t0:.2f}s")
    print(f"Methods: {module.method_names()}")

    # --- Processor / tokenizer ---
    print(f"Loading processor from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    config = AutoConfig.from_pretrained(args.model_id)
    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"Image token id: {image_token_id}")

    # --- Preprocess ---
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].squeeze(0)
    pixel_values = inputs["pixel_values"].to(torch.float32)
    image_grid_thw = inputs.get("image_grid_thw", None)
    print(f"input_ids shape: {input_ids.shape}")
    print(f"pixel_values shape: {pixel_values.shape}")

    img_start, img_end = find_image_token_span(input_ids, image_token_id)
    if img_start is None:
        print("ERROR: No image tokens found in input_ids.")
        return 1
    print(
        f"Image tokens at [{img_start}, {img_end}) ({img_end - img_start} tokens)"
    )

    # --- Vision encoder (PyTorch eager) ---
    image_embeds = run_vision_encoder_eager(
        args.model_id, pixel_values, image_grid_thw
    )
    print(f"Image embeddings shape: {image_embeds.shape}")

    # --- Prefill (ExecuTorch PTE) ---
    t_start = time.perf_counter()
    pos = 0
    logits = None

    if img_start > 0:
        logits, pos = prefill_text(module, input_ids[:img_start], pos)

    logits, pos = prefill_image_embeds(module, image_embeds, pos)

    if img_end < len(input_ids):
        logits, pos = prefill_text(module, input_ids[img_end:], pos)

    t_prefill = time.perf_counter()
    print(f"Prefill done: {pos} tokens in {t_prefill - t_start:.2f}s")

    # --- Decode ---
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    text = tokenizer.decode([next_token], skip_special_tokens=False)
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    print("Response: ", end="", flush=True)
    print(text, end="", flush=True)

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for tok_str in ["<|endoftext|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid != tokenizer.unk_token_id:
            eos_ids.add(tid)

    generated_count = 1
    for _ in range(args.max_new_tokens - 1):
        if next_token in eos_ids:
            break

        logits, pos = decode_one(module, next_token, pos)

        if args.temperature <= 0:
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        else:
            probs = torch.softmax(logits[:, -1, :] / args.temperature, dim=-1)
            next_token = torch.multinomial(probs.squeeze(0), 1).item()

        text = tokenizer.decode([next_token], skip_special_tokens=False)
        if next_token not in eos_ids:
            print(text, end="", flush=True)
        generated_count += 1

    t_end = time.perf_counter()
    gen_time = t_end - t_prefill
    print()
    print("-" * 50)
    print(f"Prompt tokens:    {pos - generated_count}")
    print(f"Generated tokens: {generated_count}")
    print(f"Prefill time:     {t_prefill - t_start:.3f}s")
    if gen_time > 0 and generated_count > 1:
        print(
            f"Decode rate:      {(generated_count - 1) / gen_time:.2f} tokens/sec"
        )
    print(f"Total time:       {t_end - t_start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
