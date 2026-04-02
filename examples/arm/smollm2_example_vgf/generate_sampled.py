#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Greedy/temperature sampling using cmake-out-vkml/executor_runner."""

import argparse
import secrets
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from pytorch_tokenizers import get_tokenizer  # type: ignore[import-untyped]


class RunnerSession:
    """Manage one executor_runner instance for prompt-by-prompt sampling.

    This wrapper hides the temporary input/output files expected by
    `executor_runner` and optionally keeps the runner process alive across
    decoding steps. Callers use `run()` with a token window and receive the
    raw logits produced for that step.
    """

    def __init__(
        self,
        runner: str,
        pte: str,
        extra_args: List[str],
        *,
        persistent: bool,
    ) -> None:
        self._runner = runner
        self._pte = pte
        self._extra_args = extra_args
        self._persistent = persistent

        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._tmpdir_path: Optional[Path] = None
        self._input_path: Optional[Path] = None
        self._output_prefix: Optional[Path] = None

        self._proc: Optional[subprocess.Popen[str]] = None
        self._recent_stdout: List[str] = []

        self._init_paths()
        if self._persistent:
            self._start_server()

    def _init_paths(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmpdir_path = Path(self._tmpdir.name)
        self._input_path = self._tmpdir_path / "tokens.bin"
        self._output_prefix = self._tmpdir_path / "logits"

    def _base_cmd(self) -> List[str]:
        assert self._input_path is not None
        assert self._output_prefix is not None
        return [
            self._runner,
            "--model_path",
            self._pte,
            "--inputs",
            str(self._input_path),
            "--output_file",
            str(self._output_prefix),
        ] + self._extra_args

    def _start_server(self) -> None:
        cmd = self._base_cmd() + ["--server_mode=true"]
        self._proc = subprocess.Popen(  # nosec B603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self) -> "RunnerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def _read_logits(self) -> np.ndarray:
        assert self._output_prefix is not None
        output_path = self._output_prefix.with_name(self._output_prefix.name + "-0.bin")
        return np.fromfile(output_path, dtype=np.float32)

    def _output_path(self) -> Path:
        assert self._output_prefix is not None
        return self._output_prefix.with_name(self._output_prefix.name + "-0.bin")

    def _check_proc(self) -> None:
        if self._proc is None:
            return
        rc = self._proc.poll()
        if rc is not None:
            tail = "".join(self._recent_stdout[-200:])
            # Drain remaining stdout (process has exited).
            extra = ""
            if self._proc.stdout is not None:
                try:
                    extra = self._proc.stdout.read() or ""
                except Exception:
                    extra = ""
            msg = f"executor_runner exited unexpectedly (rc={rc})"
            if tail.strip() or extra.strip():
                msg += "\n\n[executor_runner stdout tail]\n" + tail + extra
            raise RuntimeError(msg)

    def run(self, tokens: np.ndarray) -> np.ndarray:
        """Run executor_runner once and return output logits as float32."""

        assert self._input_path is not None
        tokens.tofile(self._input_path)

        if not self._persistent:
            cmd = self._base_cmd()
            proc = subprocess.run(  # nosec B603
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            out = proc.stdout.decode(errors="replace")
            output_path = self._output_path()

            # Prefer checking for the output file instead of relying on stdout.
            if output_path.exists() and output_path.stat().st_size > 0:
                return self._read_logits()

            rc = proc.returncode
            rc_msg = f"rc={rc}"
            if rc < 0:
                rc_msg += f" (signal {-rc})"
            raise RuntimeError(
                f"executor_runner failed ({rc_msg}).\n\n[executor_runner stdout]\n{out}"
            )

        # Persistent/server mode: trigger an execution by writing one line.
        self._check_proc()
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        self._proc.stdin.write("go\n")
        self._proc.stdin.flush()

        # Wait for sentinel.
        while True:
            self._check_proc()
            line = self._proc.stdout.readline()
            if line == "":
                # EOF
                self._check_proc()
                raise RuntimeError("executor_runner stdout closed unexpectedly")
            self._recent_stdout.append(line)
            if len(self._recent_stdout) > 400:
                self._recent_stdout = self._recent_stdout[-400:]
            if "SERVER MODE DONE" in line:
                break
        return self._read_logits()


def prepare_input(
    ids: List[int],
    window: int,
    pad_id: int,
    *,
    pad_left: bool,
    input_dtype: np.dtype,
) -> np.ndarray:
    """Prepare input token array of shape (1, window).

    For exports that only return last-token logits, the model always uses the
    last token position, so we typically left-pad to keep the newest tokens at
    the end.

    For exports that return full logits `[B, S, V]`, right-padding + selecting
    the logits row at `last_valid_token_pos` avoids pads affecting attention.
    """

    ids = ids[-window:]
    if len(ids) < window:
        pad = [pad_id] * (window - len(ids))
        ids = pad + ids if pad_left else ids + pad
    return np.array(ids, dtype=input_dtype).reshape(1, -1)


def sample_token(logits: np.ndarray, temperature: float) -> int:
    return sample_token_topk_topp(
        logits,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
    )


def apply_repetition_penalty(
    logits: np.ndarray, generated_ids: List[int], penalty: float
) -> np.ndarray:
    """Apply a repetition penalty in-place and return logits.

    This follows the common approach used by HF generation:
    - if logit > 0: logit /= penalty
    - else:         logit *= penalty

    """

    if penalty is None or penalty <= 1.0:
        return logits

    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if 0 <= token_id < logits.shape[0]:
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def sample_token_topk_topp(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample one token id from a logits vector.

    Args:
        logits: One-dimensional logits for the current decoding step.
        temperature: Sampling temperature. Non-positive values disable
            sampling and fall back to argmax.
        top_k: Number of highest-probability tokens to keep before sampling.
            Use 0 to disable top-k filtering.
        top_p: Cumulative probability threshold for nucleus sampling.

    Returns:
        The sampled token id as an integer index into `logits`.
    """

    if temperature <= 0:
        return int(np.argmax(logits))

    top_k = int(top_k)
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    if top_p <= 0 or top_p > 1.0:
        raise ValueError("top_p must be in (0, 1]")

    # Temperature scaling
    z = logits / temperature

    # Top-k filtering
    if top_k > 0 and top_k < z.size:
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z < kth, -np.inf, z)

    # Convert to probabilities
    z = z - np.max(z)
    probs = np.exp(z)
    probs_sum = probs.sum()
    if not np.isfinite(probs_sum) or probs_sum <= 0:
        # Degenerate distribution (e.g. all -inf): fall back to argmax.
        return int(np.argmax(logits))
    probs /= probs_sum

    # Top-p (nucleus) filtering
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


def topk_tokens(logits: np.ndarray, k: int) -> List[int]:
    """Return indices of the top-k logits.

    Uses `argpartition` to avoid sorting the full vocab.
    """

    k = int(k)
    if k <= 0:
        return []
    if k >= logits.size:
        # Fallback: full sort.
        return np.argsort(-logits).tolist()
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    return idx.tolist()


def build_prompt_list(
    *,
    prompt: str,
    prompt_file: Optional[Path],
    prompt_all: bool,
    prompt_random: bool,
    prompt_index: int,
    prompt_limit: Optional[int],
) -> List[str]:
    """Build the list of prompts to evaluate from CLI prompt inputs.

    Args:
        prompt: Inline prompt text used when no prompt file is provided.
        prompt_file: Optional text file containing one prompt per non-empty
            line.
        prompt_all: Whether to return all prompts from `prompt_file`.
        prompt_random: Whether to choose one random prompt from `prompt_file`.
        prompt_index: Index of the prompt to select from `prompt_file` when
            neither `prompt_all` nor `prompt_random` is used.
        prompt_limit: Optional cap on how many prompts to load from
            `prompt_file`.

    Returns:
        A list of prompt strings to feed into generation. The list contains
        either one selected prompt or the full filtered prompt file contents.
    """

    if prompt_all and prompt_file is None:
        raise ValueError("--prompt-all requires --prompt-file/--prompts-file")

    prompts: List[str]
    if prompt_file is not None:
        lines = prompt_file.read_text(encoding="utf-8").splitlines()
        prompts = [line for line in lines if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {prompt_file}")
        if prompt_limit is not None:
            if prompt_limit < 0:
                raise ValueError("--prompt-limit must be >= 0")
            prompts = prompts[:prompt_limit]
    else:
        prompts = [prompt]

    if prompt_all:
        return prompts

    if prompt_file is None:
        return [prompt]

    if prompt_random:
        return [secrets.choice(prompts)]

    if prompt_index < 0 or prompt_index >= len(prompts):
        raise ValueError(
            f"--prompt-index {prompt_index} out of range for {prompt_file} (0..{len(prompts)-1})"
        )
    return [prompts[prompt_index]]


def select_last_token_logits(
    *,
    logits: np.ndarray,
    vocab_size: Optional[int],
    window: int,
    use_full_logits: bool,
    valid_len: int,
) -> tuple[np.ndarray, Optional[int]]:
    """Select the logits row used for sampling and infer vocab size.

    Args:
        logits: Flat runner output containing either `[vocab]` logits or
            `[window * vocab]` logits for the full context window.
        vocab_size: Optional expected vocabulary size from the CLI.
        window: Decode window size used for the runner input.
        use_full_logits: Whether the export is expected to return logits for
            every position in the window.
        valid_len: Number of non-padding tokens in the current input window.

    Returns:
        A tuple of `(logits_row, inferred_vocab_size)`, where `logits_row` is
        the one-dimensional logits array for the next-token decision and
        `inferred_vocab_size` is the resolved vocab size when it can be
        inferred from the output shape.
    """

    logits_0: np.ndarray
    inferred_vocab_size: Optional[int] = None

    if vocab_size is not None and logits.size % vocab_size == 0:
        inferred_vocab_size = int(vocab_size)
        logits_2d = logits.reshape(-1, inferred_vocab_size)
        if use_full_logits and logits_2d.shape[0] == window:
            if valid_len <= 0:
                raise RuntimeError("No valid tokens to score")
            logits_0 = logits_2d[valid_len - 1]
        else:
            logits_0 = logits_2d[-1]
        return logits_0, inferred_vocab_size

    if window > 0 and logits.size % window == 0:
        inferred_vocab_size = int(logits.size // window)

        # Heuristic: treat as full logits when vocab is plausibly large.
        if inferred_vocab_size >= 1024:
            logits_2d = logits.reshape(window, inferred_vocab_size)
            if use_full_logits:
                if valid_len <= 0:
                    raise RuntimeError("No valid tokens to score")
                logits_0 = logits_2d[valid_len - 1]
            else:
                logits_0 = logits_2d[-1]
            return logits_0, inferred_vocab_size

    logits_0 = logits.reshape(1, -1)[0]
    return logits_0, inferred_vocab_size


def print_topk_candidates(
    *,
    logits: np.ndarray,
    tokenizer,
    step: int,
    k: int = 5,
) -> None:
    topk = topk_tokens(logits, k)
    print("\n--- Step", step, f"Top-{k} candidates ---")
    for idx in topk:
        print(f"{idx:5d} | {logits[idx]:8.4f} | {tokenizer.decode_token(int(idx))}")


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


def run_one_prompt(
    *,
    runner: RunnerSession,
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
    input_dtype: np.dtype,
    save_generations_path: Optional[Path],
    topk_print: bool,
) -> None:
    """Run autoregressive generation for one prompt and emit decoded text.

    Args:
        runner: Session used to execute the exported model.
        tokenizer: Tokenizer providing encode/decode helpers for the model.
        prompt: Input prompt text to seed generation.
        prompt_no: Prompt index used in logging and saved output.
        vocab_size: Optional expected vocabulary size for output decoding.
        pad_id: Token id used to pad the decode window.
        eos_id: Token id that terminates generation when sampled.
        window: Token window size passed to the exported model.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature passed to `sample_token_topk_topp`.
        top_k: Top-k sampling parameter.
        top_p: Top-p sampling parameter.
        repetition_penalty: Penalty applied to previously generated tokens.
        use_full_logits: Whether runner outputs logits for every position in
            the input window.
        save_generations_path: Optional append-only text file for completed
            generations.
        topk_print: Whether to print the top-k candidates at each step.

    Returns:
        None. The function prints streamed generation output and optionally
        appends the final decoded text to `save_generations_path`.
    """

    ids = tokenizer.encode(prompt, bos=True, eos=False)
    print(
        f"\n==================== Prompt {prompt_no} ====================\n{prompt}",
        end="",
        flush=True,
    )

    for step in range(max_new_tokens):
        # When we have full logits, right-pad so pad tokens are *after* the
        # valid prefix and can't be attended to by causal masking.
        pad_left = not use_full_logits
        window_tokens = prepare_input(
            ids,
            window,
            pad_id,
            pad_left=pad_left,
            input_dtype=input_dtype,
        )
        valid_len = min(len(ids), window)
        logits = runner.run(window_tokens)

        # Decode output shape.
        # - If the export produced full logits, executor_runner writes a flat
        #   `[window * vocab]` float array.
        # - If it produced last-token logits, it's `[vocab]`.
        logits_0, inferred_vocab_size = select_last_token_logits(
            logits=logits,
            vocab_size=vocab_size,
            window=window,
            use_full_logits=use_full_logits,
            valid_len=valid_len,
        )

        if topk_print:
            print_topk_candidates(
                logits=logits_0,
                tokenizer=tokenizer,
                step=step,
                k=5,
            )

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
        if inferred_vocab_size is not None and next_id >= inferred_vocab_size:
            raise RuntimeError(
                f"Sampled token id {next_id} out of inferred vocab_size {inferred_vocab_size}. "
                "This usually indicates a logits-shape mismatch."
            )
        ids.append(next_id)
        token_text = tokenizer.decode_token(next_id)
        print("Chosen token:", token_text)
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
        description="Temperature sampling via executor_runner"
    )
    parser.add_argument("--runner", default="cmake-out-vkml/executor_runner")
    parser.add_argument("--pte", default="smollm2_vgf_calibrated.pte")
    parser.add_argument("--tokenizer", default="data/tokenizers/smollm2/tokenizer.json")
    parser.add_argument(
        "--persistent-runner",
        action="store_true",
        help="Keep executor_runner alive using --server_mode.",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Single prompt used when no prompt file is provided.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Newline-delimited prompts file. Overrides --prompt.",
    )
    # Back-compat: older flag name.
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Alias for --prompt-file.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="0-based index into non-empty lines of --prompt-file.",
    )
    parser.add_argument(
        "--prompt-random",
        action="store_true",
        help="Select a random prompt from --prompt-file.",
    )
    parser.add_argument(
        "--prompt-all",
        action="store_true",
        help="Run generation for every non-empty line in --prompt-file.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Limit the number of prompts used from --prompt-file.",
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
        "--full-logits",
        action="store_true",
        help="Assume the export produces full logits and select the last valid token row.",
    )
    parser.add_argument(
        "--input-dtype",
        choices=("int32", "int64"),
        default="int32",
        help="Token input dtype. Use int64 only for replaying older exports.",
    )
    parser.add_argument(
        "--save-generations",
        type=Path,
        default=None,
        help="Append prompt + final decoded generation to this text file.",
    )
    parser.add_argument(
        "--no-topk-print",
        action="store_true",
        help="Disable per-step top-k printing (reduces CPU usage).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed Python and NumPy RNGs for reproducible sampling.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability mass (1.0 disables).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Top-k sampling cutoff (0 disables).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1.0 discourages repeats).",
    )
    parser.add_argument("--runner-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    np.random.seed(args.seed)

    tokenizer = get_tokenizer(args.tokenizer)
    vocab_size = getattr(tokenizer, "n_words", None)
    pad_id = getattr(tokenizer, "pad_id", getattr(tokenizer, "eos_id", 0))
    eos_id = getattr(tokenizer, "eos_id", pad_id)
    input_dtype = np.dtype(np.int64 if args.input_dtype == "int64" else np.int32)

    prompt_file = (
        args.prompt_file if args.prompt_file is not None else args.prompts_file
    )
    prompts = build_prompt_list(
        prompt=args.prompt,
        prompt_file=prompt_file,
        prompt_all=args.prompt_all,
        prompt_random=args.prompt_random,
        prompt_index=args.prompt_index,
        prompt_limit=args.prompt_limit,
    )

    with RunnerSession(
        args.runner,
        args.pte,
        args.runner_args,
        persistent=args.persistent_runner,
    ) as runner:
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
                input_dtype=input_dtype,
                save_generations_path=args.save_generations,
                topk_print=not args.no_topk_print,
            )


if __name__ == "__main__":
    main()
