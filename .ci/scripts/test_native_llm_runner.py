# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare GGML LLM runner prefill and decoding with a tiny eager HF Llama,
including a prompt longer than the method's token input."""

import argparse
import base64
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from executorch.backends.native.examples.hf_llm.export_hf_llm import LastTokenLM
from executorch.exir.native import to_native
from transformers import LlamaConfig, LlamaForCausalLM


def export_package(
    model: LlamaForCausalLM, max_seq_length: int, max_tokens: int, path: Path
) -> None:
    """A last-token package whose method takes at most `max_tokens` tokens."""
    wrapper = LastTokenLM(model, max_seq_length).eval()
    seq = torch.export.Dim("seq", min=1, max=max_tokens)
    exported = torch.export.export(
        wrapper,
        (torch.zeros((1, 6), dtype=torch.long), torch.arange(6)),
        dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
        strict=True,
    )
    to_native(exported).save(str(path))


def check_runner(
    runner: Path,
    model: LlamaForCausalLM,
    package: Path,
    tokenizer: Path,
    prompt: str,
    max_seq_length: int,
    max_new_tokens: int,
) -> None:
    """Checks the runner's greedy tokens after `prompt` against eager HF."""
    config = model.config
    unreachable_eos_id = config.vocab_size
    tokens = [config.bos_token_id, *prompt.encode()]
    expected = []
    # Recompute the full prefix without a cache for an independent reference.
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([tokens]), use_cache=False).logits
        token = int(logits[0, -1].argmax())
        expected.append(token)
        tokens.append(token)

    result = subprocess.run(
        [
            str(runner.resolve()),
            "--device=cpu",
            "--cpu_threads=4",
            f"--package={package}",
            f"--tokenizer={tokenizer}",
            f"--prompt={prompt}",
            f"--eos_ids={unreachable_eos_id}",
            f"--max_new_tokens={max_new_tokens}",
            f"--max_seq_length={max_seq_length}",
            "--print_token_ids",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    print(result.stdout, end="")
    print(result.stderr, end="")
    result.check_returncode()
    match = re.search(r"^token_ids: ([0-9,]+)$", result.stdout, re.MULTILINE)
    if match is None:
        raise AssertionError("Runner did not report generated token IDs")
    actual = [int(token) for token in match[1].split(",")]
    if actual != expected:
        raise AssertionError(f"Generated tokens {actual} != eager {expected}")
    print(f"{package.name}: matched all {max_new_tokens} eager token IDs")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(4)
    max_seq_length = 48
    max_new_tokens = 8
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_seq_length,
        bos_token_id=256,
        eos_token_id=257,
        attn_implementation="sdpa",
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokenizer = root / "tokenizer.model"
        # The byte vocabulary puts Llama's BOS at 256, followed by 255 specials.
        tokenizer.write_text(
            "".join(
                f"{base64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256)
            )
        )
        full = root / "llama.ptn"
        export_package(model, max_seq_length, max_seq_length - 1, full)
        # A 20-token prompt prefills in three calls of at most 7 tokens.
        chunked = root / "llama_chunked.ptn"
        export_package(model, max_seq_length, 7, chunked)
        for package, prompt in ((full, "abc"), (chunked, "the quick brown fox")):
            check_runner(
                args.runner,
                model,
                package,
                tokenizer,
                prompt,
                max_seq_length,
                max_new_tokens,
            )


if __name__ == "__main__":
    main()
