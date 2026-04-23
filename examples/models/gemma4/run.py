#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an exported Gemma 4 .pte from Python via the ExecuTorch runtime.

Mirrors the C++ runner (``main.cpp`` / ``cmake-out/.../gemma4_runner``) but
without needing to build C++. Useful for quick parity checks against
``inference.py`` (HF eager).

Currently supports text-only prompts. Image and audio paths are TODO — for
multimodal use the C++ runner today.

Usage:
    python -m executorch.examples.models.gemma4.run \
        --pte /tmp/gemma4_multimodal_v11.pte \
        --tokenizer ~/models/gemma-4-E2B-it/tokenizer.json \
        --prompt "What is the capital of France?" \
        --max-new-tokens 30
"""
from __future__ import annotations

import argparse
import time

import torch
from executorch.runtime import Runtime, Verification


def encode_chat(tokenizer, prompt: str):
    text = (
        "<bos><|turn>user\n"
        + prompt
        + "<turn|>\n<|turn>model\n"
    )
    out = tokenizer.encode(text, add_special_tokens=False)
    return out.ids if hasattr(out, "ids") else out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pte", required=True, help="Path to gemma4 .pte")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to tokenizer.json (HF format)")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    print(f"Loading tokenizer {args.tokenizer}...")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    print(f"Loading {args.pte}...")
    runtime = Runtime.get()
    program = runtime.load_program(args.pte, verification=Verification.Minimal)
    token_emb = program.load_method("token_embedding")
    text_dec = program.load_method("text_decoder")

    has_pli = len(text_dec.method_meta.input_tensor_meta_list) >= 3
    print(f"text_decoder PLI input: {'enabled (3-input)' if has_pli else 'disabled (2-input)'}")

    ids = encode_chat(tokenizer, args.prompt)
    print(f"Prompt: {args.prompt!r}  -> {len(ids)} tokens")
    ids_t = torch.tensor([ids], dtype=torch.long)

    # Prefill
    t0 = time.time()
    embeds = token_emb.execute([ids_t])[0]
    pos = torch.tensor([0], dtype=torch.long)
    if has_pli:
        logits = text_dec.execute([embeds, pos, ids_t])[0]
    else:
        logits = text_dec.execute([embeds, pos])[0]
    last = logits[..., -1, :] if logits.dim() == 3 else logits[0]
    nxt = int(torch.argmax(last).item())
    prefill_dt = time.time() - t0

    eos_ids = {1, 106, 50}
    gen = []
    cur = nxt
    pos_val = len(ids)
    t0 = time.time()
    for _ in range(args.max_new_tokens):
        if cur in eos_ids:
            break
        gen.append(cur)
        cur_t = torch.tensor([[cur]], dtype=torch.long)
        e = token_emb.execute([cur_t])[0]
        p = torch.tensor([pos_val], dtype=torch.long)
        if has_pli:
            l = text_dec.execute([e, p, cur_t])[0]
        else:
            l = text_dec.execute([e, p])[0]
        last = l[..., -1, :] if l.dim() == 3 else l[0]
        cur = int(torch.argmax(last).item())
        pos_val += 1
    decode_dt = time.time() - t0

    text = tokenizer.decode(gen)
    n = len(gen)
    print(f"Output: {text!r}")
    print(f"prefill: {prefill_dt*1000:.0f} ms ({len(ids)} tokens, {len(ids)/prefill_dt:.0f} tok/s)")
    print(f"decode:  {decode_dt*1000:.0f} ms ({n} tokens, {n/max(decode_dt,1e-6):.1f} tok/s)")


if __name__ == "__main__":
    main()
