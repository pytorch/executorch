#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager-mode HuggingFace reference for Gemma 4.

Used as ground truth for the parity tests under ``tests/`` and as a quick
sanity check that the HF checkpoint produces the expected canonical output
("The capital of France is **Paris**.") on a given environment.

Usage:
    python -m executorch.examples.models.gemma4.inference \
        --hf-model ~/models/gemma-4-E2B-it \
        --prompt "What is the capital of France?" \
        --max-new-tokens 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def encode_chat(tokenizer, prompt: str) -> torch.Tensor:
    text = (
        "<bos><|turn>user\n"
        + prompt
        + "<turn|>\n<|turn>model\n"
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True, help="HF Gemma 4 model dir")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--variant", default="e2b", choices=["e2b", "e4b"])
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16"])
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"Loading {args.hf_model} ({args.variant}, {args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=dtype, trust_remote_code=True
    ).eval()

    ids = encode_chat(tokenizer, args.prompt)
    print(f"Prompt: {args.prompt!r}")
    print(f"  -> {ids.shape[1]} tokens: {ids[0].tolist()}")

    print(f"Generating up to {args.max_new_tokens} tokens (greedy)...")
    eos_ids = [1, 106, 50]  # <eos>, <turn|>, token 50
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=eos_ids[0],
        )
    dt = time.time() - t0
    new_tokens = out[0, ids.shape[1]:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    n = len(new_tokens)
    print(f"Generated {n} tokens in {dt:.2f}s ({n/dt:.1f} tok/s)")
    print(f"Output: {text!r}")


if __name__ == "__main__":
    main()
