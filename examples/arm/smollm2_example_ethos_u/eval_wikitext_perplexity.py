#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from generate_sampled import (  # type: ignore[import-not-found]
    FvpRunnerSession,
    prepare_input,
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
            "The 'datasets' package is required to build Wikitext prompts."
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
    """Build fixed-length prompts from Wikitext.

    The evaluator compares runners with a fixed inference window, so this helper
    trims each accepted prompt to a bounded token count instead of feeding
    arbitrarily long Wikitext paragraphs into the runtime.

    Args:
        tokenizer (Any): Tokenizer used to measure and decode prompts.
        split (str): Wikitext split to load.
        num_prompts (int): Number of prompts to build.
        min_prompt_tokens (int): Minimum token count before accepting a prompt.
        max_prompt_tokens (int): Maximum token count retained for each prompt.

    Returns:
        List[str]: Prompt strings ready to save or evaluate.

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
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def read_prompts(path: Path, limit: int) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if len(prompts) < limit:
        raise RuntimeError(
            f"Prompt file {path} only contains {len(prompts)} prompts; need {limit}."
        )
    return prompts[:limit]


def token_nll(logits: np.ndarray, target_id: int) -> float:
    max_logit = float(np.max(logits))
    shifted = logits - max_logit
    log_denom = max_logit + math.log(float(np.exp(shifted).sum()))
    return log_denom - float(logits[target_id])


def reshape_full_logits(*, logits: np.ndarray, window: int) -> np.ndarray:
    """Reshape flat FVP output into `[window, vocab]` full-logits rows."""
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
    runner: FvpRunnerSession,
    tokenizer,
    prompt: str,
    window: int,
    pad_id: int,
    max_tokens_per_prompt: int,
) -> Tuple[float, int]:
    """Score one prompt with a fixed-window full-logits runner.

    The deployed runner expects exactly `window` token slots every time. For
    perplexity we therefore right-pad shorter prompts so the valid prompt tokens
    remain at the front of the causal window and each logits row still lines up
    with the matching target token.

    Args:
        runner (FvpRunnerSession): Active FVP session.
        tokenizer (Any): Tokenizer used for encoding.
        prompt (str): Prompt text to score.
        window (int): Fixed inference window.
        pad_id (int): Token id used for right padding.
        max_tokens_per_prompt (int): Optional prompt length cap.

    Returns:
        Tuple[float, int]: Total negative log likelihood and scored token count.

    """
    token_ids = tokenizer.encode(prompt, bos=True, eos=False)
    if max_tokens_per_prompt > 0:
        token_ids = token_ids[:max_tokens_per_prompt]
    if len(token_ids) < 2:
        return 0.0, 0

    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    input_ids = input_ids[-window:]
    target_ids = target_ids[-len(input_ids) :]

    # Right padding keeps the real prompt tokens at the front of the window, so
    # row `i` in the full-logits output still corresponds to target token `i`.
    window_tokens = prepare_input(
        input_ids,
        window,
        pad_id,
        pad_left=False,
    )
    valid_len = min(len(input_ids), window)
    logits = runner.run(window_tokens)
    logits_2d = reshape_full_logits(logits=logits, window=window)

    total_nll = 0.0
    for row_index, target_id in enumerate(target_ids[:valid_len]):
        if target_id >= logits_2d.shape[1]:
            raise RuntimeError(
                f"Target token id {target_id} out of inferred vocab size {logits_2d.shape[1]}."
            )
        total_nll += token_nll(logits_2d[row_index], target_id)
    return total_nll, valid_len


def eval_model_ppl(
    *,
    name: str,
    fvp: str,
    runner: str,
    pte: Optional[str],
    tokenizer,
    prompts: List[str],
    window: int,
    pad_id: int,
    max_tokens_per_prompt: int,
    timeout: int,
) -> float:
    """Run FVP for each prompt and return perplexity for one runner."""
    total_nll = 0.0
    total_tokens = 0
    with FvpRunnerSession(fvp, runner, pte, timeout) as session:
        for idx, prompt in enumerate(prompts, start=1):
            print(f"[eval] {name} prompt {idx}/{len(prompts)}")
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
        raise RuntimeError(f"No prompt tokens were scored for {name}.")
    return math.exp(total_nll / total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Wikitext prompts and compare SmolLM2 Ethos-U perplexity."
    )
    parser.add_argument(
        "--fvp",
        default="examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320",
    )
    parser.add_argument(
        "--runner-w8a8",
        required=True,
        help="Semihosting runner ELF for the w8a8 full-logits export.",
    )
    parser.add_argument(
        "--runner-w8a16",
        required=True,
        help="Semihosting runner ELF for the w8a16 full-logits export.",
    )
    parser.add_argument(
        "--pte-w8a8",
        default=None,
        help="Optional external PTE for w8a8. Omit when the runner embeds the PTE.",
    )
    parser.add_argument(
        "--pte-w8a16",
        default=None,
        help="Optional external PTE for w8a16. Omit when the runner embeds the PTE.",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/tokenizers/smollm2/tokenizer.json",
        help="Tokenizer JSON used for prompt building and scoring.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("examples/arm/smollm2_example_ethos_u/wikitext_prompts_100.txt"),
        help="Prompt cache file. Reused unless --refresh-prompts is set.",
    )
    parser.add_argument(
        "--wikitext-split",
        default="test",
        help="Wikitext split used when rebuilding prompts.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="How many prompts to build into --prompts-file.",
    )
    parser.add_argument(
        "--ppl-prompts",
        type=int,
        default=10,
        help="How many cached prompts to score when computing perplexity.",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=8,
        help="Discard Wikitext samples shorter than this token count.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=8,
        help="Trim accepted prompts to at most this many tokens.",
    )
    parser.add_argument(
        "--max-tokens-per-prompt",
        type=int,
        default=8,
        help="Cap scored tokens per prompt. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="Fixed runner window. Must match the exported model shape.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="FVP time limit in seconds for each runner invocation.",
    )
    parser.add_argument(
        "--refresh-prompts",
        action="store_true",
        help="Rebuild prompts even if --prompts-file already exists.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer)
    pad_id = getattr(tokenizer, "pad_id", getattr(tokenizer, "eos_id", 0))

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

    results = {
        "w8a8": eval_model_ppl(
            name="w8a8",
            fvp=args.fvp,
            runner=args.runner_w8a8,
            pte=args.pte_w8a8,
            tokenizer=tokenizer,
            prompts=prompts,
            window=args.window,
            pad_id=pad_id,
            max_tokens_per_prompt=args.max_tokens_per_prompt,
            timeout=args.timeout,
        ),
        "w8a16": eval_model_ppl(
            name="w8a16",
            fvp=args.fvp,
            runner=args.runner_w8a16,
            pte=args.pte_w8a16,
            tokenizer=tokenizer,
            prompts=prompts,
            window=args.window,
            pad_id=pad_id,
            max_tokens_per_prompt=args.max_tokens_per_prompt,
            timeout=args.timeout,
        ),
    }

    print("\n=== Perplexity summary ===")
    for name, ppl in results.items():
        print(f"{name:8s}: {ppl:.4f}")


if __name__ == "__main__":
    main()
