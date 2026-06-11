#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Build Wikitext prompts and compare perplexity across SmolLM2 VGF exports."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_sampled import (  # type: ignore[import-not-found]
    prepare_input,
    RunnerSession,
)
from pytorch_tokenizers import (  # type: ignore[import-not-found, import-untyped]
    get_tokenizer,
)


def _load_wikitext_lines(split: str) -> Iterable[str]:
    try:
        from datasets import (  # type: ignore[import-not-found, import-untyped]
            load_dataset,
        )
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it in the active environment "
            "to download Wikitext prompts."
        ) from exc

    dataset = load_dataset(  # nosec B615
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
    )
    for entry in dataset["text"]:
        yield entry


def build_prompts(
    *,
    tokenizer,
    split: str,
    num_prompts: int,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
) -> List[str]:
    """Build reusable Wikitext prompts within the requested token range.

    The raw Wikitext split contains headings, blank lines, and short fragments.
    This function joins adjacent content lines until enough tokens are
    available, truncates each prompt to `max_prompt_tokens`, and returns exactly
    `num_prompts` decoded prompt strings.
    """

    prompts: List[str] = []
    current_parts: List[str] = []

    for raw_line in _load_wikitext_lines(split):
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        if line.startswith("=") and line.endswith("="):
            continue

        current_parts.append(line)
        candidate = " ".join(current_parts)
        token_ids = tokenizer.encode(candidate, bos=False, eos=False)

        if len(token_ids) < min_prompt_tokens:
            continue

        token_ids = token_ids[:max_prompt_tokens]
        prompts.append(tokenizer.decode(token_ids).strip())
        current_parts = []

        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"Only built {len(prompts)} prompts from Wikitext; requested {num_prompts}."
        )

    return prompts


def write_prompts(path: Path, prompts: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(prompts) + "\n"
    path.write_text(text, encoding="utf-8")


def read_prompts(path: Path, limit: int) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if len(prompts) < limit:
        raise RuntimeError(
            f"Prompt file {path} only contains {len(prompts)} prompts; need {limit}."
        )
    return prompts[:limit]


def reshape_full_logits(*, logits: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be > 0")
    if logits.size % window != 0:
        raise RuntimeError(
            f"Expected full-logits output divisible by window={window}, got size={logits.size}."
        )
    vocab_size = logits.size // window
    if vocab_size <= 0:
        raise RuntimeError(f"Invalid inferred vocab size {vocab_size}.")
    return logits.reshape(window, vocab_size)


def eval_prompt_nll(
    *,
    runner: RunnerSession,
    tokenizer,
    prompt: str,
    window: int,
    pad_id: int,
    max_tokens_per_prompt: int,
) -> Tuple[float, int]:
    token_ids = tokenizer.encode(prompt, bos=True, eos=False)
    if max_tokens_per_prompt > 0:
        token_ids = token_ids[:max_tokens_per_prompt]

    if len(token_ids) < 2:
        return 0.0, 0

    # Score the entire prompt with a single full-logits runner invocation.
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    window_tokens = prepare_input(
        input_ids,
        window,
        pad_id,
        pad_left=False,
        input_dtype=np.int32,
    )
    valid_len = min(len(input_ids), window)
    logits = runner.run(window_tokens)
    logits_2d = reshape_full_logits(logits=logits, window=window)

    # prepare_input keeps the scored tokens at positions [0, valid_len).
    rows = logits_2d[:valid_len]
    targets = np.asarray(target_ids[-valid_len:], dtype=np.int64)
    if targets.size and (targets.min() < 0 or targets.max() >= rows.shape[1]):
        raise RuntimeError(
            f"Target token id out of inferred vocab size {rows.shape[1]}."
        )

    max_logits = rows.max(axis=1)
    shifted = rows - max_logits[:, None]
    log_denom = max_logits + np.log(np.exp(shifted).sum(axis=1))
    total_nll = float((log_denom - rows[np.arange(valid_len), targets]).sum())

    return total_nll, valid_len


def eval_model_ppl(
    *,
    runner: Path,
    pte: Path,
    tokenizer,
    prompts: List[str],
    window: int,
    pad_id: int,
    max_tokens_per_prompt: int,
) -> float:
    total_nll = 0.0
    total_tokens = 0

    with RunnerSession(
        runner=str(runner),
        pte=str(pte),
        extra_args=[],
        persistent=True,
    ) as session:
        for idx, prompt in enumerate(prompts, start=1):
            print(f"[eval] {pte.name} prompt {idx}/{len(prompts)}")
            prompt_nll, prompt_tokens = eval_prompt_nll(
                runner=session,
                tokenizer=tokenizer,
                prompt=prompt,
                window=window,
                pad_id=pad_id,
                max_tokens_per_prompt=max_tokens_per_prompt,
            )
            total_nll += prompt_nll
            total_tokens += prompt_tokens

    if total_tokens == 0:
        raise RuntimeError("No prompt tokens were scored.")

    return math.exp(total_nll / total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Wikitext prompts and compare SmolLM2 VGF perplexity."
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("cmake-out-vkml/executor_runner"),
        help="Path to the VKML executor_runner binary.",
    )
    parser.add_argument(
        "--pte-fp32",
        type=Path,
        required=True,
        help="Path to the FP32 full-logits SmolLM2 VGF PTE.",
    )
    parser.add_argument(
        "--pte-linear8a8w",
        type=Path,
        default=None,
        help="Path to the linear-only 8a8w full-logits SmolLM2 VGF PTE.",
    )
    parser.add_argument(
        "--pte-linear16a8w",
        type=Path,
        default=None,
        help="Path to the linear-only 16a8w full-logits SmolLM2 VGF PTE.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/smollm2/tokenizer.json"),
        help="Path to the SmolLM2 tokenizer.json file.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("examples/arm/smollm2_example_vgf/wikitext_prompts_1000.txt"),
        help="Path to the reusable Wikitext prompt file.",
    )
    parser.add_argument("--wikitext-split", default="test")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--ppl-prompts", type=int, default=100)
    parser.add_argument("--min-prompt-tokens", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument(
        "--max-tokens-per-prompt",
        type=int,
        default=64,
        help="Cap scored tokens per prompt. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--max-seq-length",
        "--window",
        dest="window",
        type=int,
        default=64,
        help="Fixed token sequence length expected by the exported model.",
    )
    parser.add_argument(
        "--refresh-prompts",
        action="store_true",
        help="Rebuild prompts even if --prompts-file already exists.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(str(args.tokenizer))
    pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)

    if args.refresh_prompts or not args.prompts_file.exists():
        prompts = build_prompts(
            tokenizer=tokenizer,
            split=args.wikitext_split,
            num_prompts=args.num_prompts,
            min_prompt_tokens=args.min_prompt_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        write_prompts(args.prompts_file, prompts)
        print(f"[saved] {args.prompts_file} ({len(prompts)} prompts)")

    prompts = read_prompts(args.prompts_file, args.ppl_prompts)
    print(f"[info] Using first {len(prompts)} prompts from {args.prompts_file}")

    ptes = {
        "fp32": args.pte_fp32,
        "linear8a8w": args.pte_linear8a8w,
        "linear16a8w": args.pte_linear16a8w,
    }
    results = {}
    for name, pte in ptes.items():
        if pte is None:
            continue
        results[name] = eval_model_ppl(
            runner=args.runner,
            pte=pte,
            tokenizer=tokenizer,
            prompts=prompts,
            window=args.window,
            pad_id=pad_id,
            max_tokens_per_prompt=args.max_tokens_per_prompt,
        )

    print("\n=== Perplexity summary ===")
    for name, ppl in results.items():
        print(f"{name:12s}: {ppl:.4f}")


if __name__ == "__main__":
    main()
