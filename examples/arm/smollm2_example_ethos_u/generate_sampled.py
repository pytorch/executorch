#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import re
import secrets
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from pytorch_tokenizers import (  # type: ignore[import-not-found, import-untyped]
    get_tokenizer,
)

FVP_ERROR_PATTERN = re.compile(
    r"(^[EF][: ].*$)|(^.*Hard fault.*$)|(^.*Assertion.*$)",
    re.MULTILINE,
)


def prepare_input(
    ids: List[int],
    window: int,
    pad_id: int,
    *,
    pad_left: bool = True,
) -> np.ndarray:
    """Pack token IDs into the fixed-shape input tensor expected by FVP."""
    ids = ids[-window:]
    if len(ids) < window:
        pad = [pad_id] * (window - len(ids))
        ids = pad + ids if pad_left else ids + pad
    return np.array(ids, dtype=np.int32).reshape(1, -1)


def sample_token_topk_topp(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    z = logits / temperature
    if top_k > 0 and top_k < z.size:
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z < kth, -np.inf, z)

    z = z - np.max(z)
    probs = np.exp(z)
    probs_sum = probs.sum()
    if not np.isfinite(probs_sum) or probs_sum <= 0:
        return int(np.argmax(logits))
    probs /= probs_sum

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumsum, top_p, side="left"))
        cutoff = max(1, cutoff + 1)
        keep = sorted_idx[:cutoff]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        filtered_sum = filtered.sum()
        if filtered_sum > 0:
            probs = filtered / filtered_sum

    return int(np.random.choice(len(probs), p=probs))


def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: List[int],
    penalty: float,
) -> np.ndarray:
    if penalty is None or penalty <= 1.0:
        return logits
    for token_id in set(generated_ids):
        if 0 <= token_id < logits.shape[0]:
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def topk_tokens(logits: np.ndarray, k: int) -> List[int]:
    if k <= 0:
        return []
    if k >= logits.size:
        return np.argsort(-logits).tolist()
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return idx.tolist()


def print_topk_candidates(logits: np.ndarray, tokenizer, step: int, k: int = 5) -> None:
    topk = topk_tokens(logits, k)
    print(f"\n--- Step {step} Top-{k} candidates ---")
    for idx in topk:
        print(f"{idx:5d} | {logits[idx]:8.4f} | {tokenizer.decode_token(int(idx))}")


def select_last_token_logits(
    *,
    logits: np.ndarray,
    vocab_size: Optional[int],
    window: int,
    use_full_logits: bool,
    valid_len: int,
) -> np.ndarray:
    """Return the logits row used to sample the next token.

    For normal generation exports the runner emits only one logits vector. For
    full-logits exports it emits one row per token position in the fixed window,
    so we select the row that corresponds to the last real prompt token.
    """
    if use_full_logits:
        if window <= 0:
            raise ValueError("window must be > 0 when --full-logits is set")
        if logits.size % window != 0:
            raise RuntimeError(
                f"Expected full-logits output divisible by window={window}, got size={logits.size}."
            )
        inferred_vocab_size = logits.size // window
        if vocab_size is not None and inferred_vocab_size < vocab_size:
            raise RuntimeError(
                f"Inferred vocab size {inferred_vocab_size} is smaller than tokenizer vocab {vocab_size}."
            )
        logits_2d = logits.reshape(window, inferred_vocab_size)
        if valid_len <= 0:
            raise RuntimeError("No valid tokens available to select last-token logits")
        logits_0 = logits_2d[valid_len - 1]
    else:
        logits_0 = logits.reshape(1, -1)[0]

    if vocab_size is not None and logits_0.shape[0] > vocab_size:
        logits_0 = logits_0[:vocab_size]
    return logits_0


def build_prompt_list(
    *,
    prompt: str,
    prompt_file: Optional[Path],
    prompt_all: bool,
    prompt_random: bool,
    prompt_index: int,
    prompt_limit: Optional[int],
) -> List[str]:
    """Resolve prompt-selection CLI flags into a concrete prompt list."""
    if prompt_all and prompt_file is None:
        raise ValueError("--prompt-all requires --prompt-file")

    if prompt_file is None:
        prompts = [prompt]
    else:
        prompts = [
            line
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if prompt_limit is not None:
            prompts = prompts[:prompt_limit]
        if not prompts:
            raise ValueError(f"No prompts found in {prompt_file}")

    if prompt_all:
        return prompts
    if prompt_file is None:
        return [prompt]
    if prompt_random:
        return [secrets.choice(prompts)]
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise ValueError(
            f"--prompt-index {prompt_index} out of range for {prompt_file} (0..{len(prompts) - 1})"
        )
    return [prompts[prompt_index]]


def append_generation(
    *,
    path: Path,
    prompt: str,
    prompt_no: int,
    decoded: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"==================== Prompt {prompt_no} ====================\n")
        f.write(prompt)
        if not prompt.endswith("\n"):
            f.write("\n")
        f.write("\n=== Generation complete ===\n")
        f.write(decoded)
        if not decoded.endswith("\n"):
            f.write("\n")


class FvpRunnerSession:
    """Manage a temporary semihosting workspace for repeated FVP runs."""

    def __init__(
        self,
        fvp: str,
        runner: str,
        pte: Optional[str],
        timeout: int,
    ) -> None:
        self._fvp = fvp
        self._runner = runner
        self._pte = pte
        self._timeout = timeout
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._tmpdir_path: Optional[Path] = None
        self._input_path: Optional[Path] = None
        self._output_prefix: Optional[Path] = None
        self._program_path: Optional[Path] = None
        self._init_paths()

    def _init_paths(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmpdir_path = Path(self._tmpdir.name)
        self._input_path = self._tmpdir_path / "i0.bin"
        self._output_prefix = self._tmpdir_path / "out"
        if self._pte is not None:
            self._program_path = self._tmpdir_path / "program.pte"
            shutil.copyfile(self._pte, self._program_path)

    def _build_command(self, cmd_line: str) -> List[str]:
        assert self._tmpdir_path is not None
        return [
            self._fvp,
            "-C",
            "mps4_board.subsystem.ethosu.num_macs=256",
            "-C",
            "mps4_board.visualisation.disable-visualisation=1",
            "-C",
            "vis_hdlcd.disable_visualisation=1",
            "-C",
            "mps4_board.telnetterminal0.start_telnet=0",
            "-C",
            "mps4_board.uart0.out_file='-'",
            "-C",
            "mps4_board.uart0.unbuffered_output=1",
            "-C",
            "mps4_board.uart0.shutdown_on_eot=1",
            "-C",
            "mps4_board.subsystem.cpu0.semihosting-enable=1",
            "-C",
            "mps4_board.subsystem.cpu0.semihosting-stack_base=0",
            "-C",
            "mps4_board.subsystem.cpu0.semihosting-heap_limit=0",
            "-C",
            f"mps4_board.subsystem.cpu0.semihosting-cwd={self._tmpdir_path}",
            "-C",
            "mps4_board.subsystem.ethosu.extra_args='--fast'",
            "-C",
            f"mps4_board.subsystem.cpu0.semihosting-cmd_line='{cmd_line}'",
            "-a",
            self._runner,
            "--timelimit",
            str(self._timeout),
        ]

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self) -> "FvpRunnerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def _run_once(self, tokens: np.ndarray) -> np.ndarray:
        assert self._tmpdir_path is not None
        assert self._input_path is not None
        assert self._output_prefix is not None
        tokens.tofile(self._input_path)

        output_path = self._output_prefix.with_name(self._output_prefix.name + "-0.bin")
        if output_path.exists():
            output_path.unlink()

        cmd_line = "executor_runner"
        if self._program_path is not None:
            cmd_line += " -m program.pte"
        cmd_line += " -o out -i i0.bin"
        proc = subprocess.run(
            self._build_command(cmd_line),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )  # nosec B603
        out = proc.stdout.decode(errors="replace")
        matches = [m.group(0).strip() for m in FVP_ERROR_PATTERN.finditer(out)]
        if (
            proc.returncode == 0
            and not matches
            and output_path.exists()
            and output_path.stat().st_size > 0
        ):
            return np.fromfile(output_path, dtype=np.float32)
        hint = ""
        if "input size (" in out and "tensor size (" in out and "mismatch" in out:
            hint = (
                "\nLikely cause: `--window` does not match the exported model input shape. "
                "For example, a seq8 export must be run with `--window 8`."
            )
        if matches:
            hint += "\nDetected FVP/runtime fault markers:\n" + "\n".join(matches)
        raise RuntimeError(
            f"FVP execution failed (rc={proc.returncode}).{hint}\n\n[FVP stdout]\n{out}"
        )

    def run(self, tokens: np.ndarray) -> np.ndarray:
        return self._run_once(tokens)


def run_one_prompt(
    *,
    runner: FvpRunnerSession,
    tokenizer,
    prompt: str,
    prompt_no: int,
    vocab_size: Optional[int],
    pad_id: int,
    eos_id: int,
    window: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    use_full_logits: bool,
    save_generations_path: Optional[Path],
    topk_print: bool,
) -> None:
    ids = tokenizer.encode(prompt, bos=True, eos=False)
    print(
        f"\n==================== Prompt {prompt_no} ====================\n{prompt}",
        end="",
        flush=True,
    )
    if not use_full_logits and len(ids) < window:
        print(
            "\n[note] Generation exports left-pad short prompts so the last real "
            "token lands in the final input slot. Full-logits exports instead "
            "keep prompt tokens left-aligned and select the last valid row, so "
            "short-prompt continuations may differ across the two artifact types.",
            flush=True,
        )
    for step in range(max_new_tokens):
        window_tokens = prepare_input(
            ids,
            window,
            pad_id,
            pad_left=not use_full_logits,
        )
        valid_len = min(len(ids), window)
        logits = runner.run(window_tokens)
        logits_0 = select_last_token_logits(
            logits=logits,
            vocab_size=vocab_size,
            window=window,
            use_full_logits=use_full_logits,
            valid_len=valid_len,
        )
        if topk_print:
            print_topk_candidates(logits_0, tokenizer, step, k=5)
        logits_0 = apply_repetition_penalty(
            logits_0.copy(),
            generated_ids=ids,
            penalty=repetition_penalty,
        )
        next_id = sample_token_topk_topp(
            logits_0,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        ids.append(next_id)
        token_text = tokenizer.decode_token(next_id)
        print(token_text, end="", flush=True)
        if next_id == eos_id:
            break
    print("\n=== Generation complete ===")
    decoded = tokenizer.decode(ids)
    print(decoded)
    if save_generations_path is not None:
        append_generation(
            path=save_generations_path,
            prompt=prompt,
            prompt_no=prompt_no,
            decoded=decoded,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompted generation on Ethos-U FVP via semihosting executor_runner"
    )
    parser.add_argument("--fvp", required=True)
    parser.add_argument(
        "--runner", required=True, help="Semihosting arm_executor_runner ELF"
    )
    parser.add_argument("--pte", default=None)
    parser.add_argument(
        "--embedded-pte",
        action="store_true",
        help="Use the PTE embedded in the runner ELF instead of passing -m program.pte.",
    )
    parser.add_argument("--tokenizer", default="data/tokenizers/smollm2/tokenizer.json")
    parser.add_argument(
        "--prompt",
        default="Once upon a time in a small village,",
        help="Single prompt string used when --prompt-file is omitted.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file with one prompt per line.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Index to read from --prompt-file when not using --prompt-random or --prompt-all.",
    )
    parser.add_argument(
        "--prompt-random",
        action="store_true",
        help="Pick one random prompt from --prompt-file.",
    )
    parser.add_argument(
        "--prompt-all",
        action="store_true",
        help="Run generation for every prompt found in --prompt-file.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Read at most this many prompts from --prompt-file before selection.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=16,
        help="Fixed input window. Must match the exported model shape.",
    )
    parser.add_argument(
        "--save-generations",
        type=Path,
        default=None,
        help="Append prompt + final decoded generation to this text file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to append after the prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling is enabled.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling threshold. Has no effect when --temperature <= 0.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Top-k cutoff. Use 0 to disable top-k filtering.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1.0 discourages repeats, including in greedy decoding).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="FVP time limit in seconds for each runner call.",
    )
    parser.add_argument(
        "--full-logits",
        action="store_true",
        help="Interpret runner output as full logits [window, vocab] and select the last valid token row.",
    )
    parser.add_argument(
        "--no-topk-print",
        action="store_true",
        help="Suppress the per-step top-5 candidate dump.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer)
    vocab_size = getattr(tokenizer, "n_words", None)
    pad_id = getattr(tokenizer, "pad_id", getattr(tokenizer, "eos_id", 0))
    eos_id = getattr(tokenizer, "eos_id", pad_id)
    prompts = build_prompt_list(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        prompt_all=args.prompt_all,
        prompt_random=args.prompt_random,
        prompt_index=args.prompt_index,
        prompt_limit=args.prompt_limit,
    )

    pte_path = None if args.embedded_pte else args.pte
    if not args.embedded_pte and pte_path is None:
        raise ValueError("--pte is required unless --embedded-pte is set")

    with FvpRunnerSession(args.fvp, args.runner, pte_path, args.timeout) as runner:
        for i, prompt in enumerate(prompts):
            run_one_prompt(
                runner=runner,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_no=i,
                vocab_size=vocab_size,
                pad_id=pad_id,
                eos_id=eos_id,
                window=args.window,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_full_logits=args.full_logits,
                save_generations_path=args.save_generations,
                topk_print=not args.no_topk_print,
            )


if __name__ == "__main__":
    main()
