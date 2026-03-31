#!/usr/bin/env python3
"""Benchmark local mlx-audio against the cached MLX session backend."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

from mlx_backend import Qwen3TTSMlxBackend


DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_REF_TEXT = "This is what my voice sounds like."


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_ref_audio() -> Path:
    return _repo_root() / "poem.wav"


def _default_prompts_path() -> Path:
    return Path(__file__).with_name("benchmark_prompts.txt")


def _load_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def _print_summary(name: str, metrics) -> None:
    avg_throughput = mean(metric.throughput_x for metric in metrics)
    avg_first_audio = mean(metric.first_audio_s for metric in metrics)
    total_audio = sum(metric.audio_s for metric in metrics)
    total_elapsed = sum(metric.elapsed_s for metric in metrics)
    total_throughput = total_audio / total_elapsed if total_elapsed > 0.0 else 0.0
    print()
    print(f"{name} summary")
    print(f"  Average throughput : {avg_throughput:.3f}x  (> 1 = faster than real-time)")
    print(f"  Total throughput   : {total_throughput:.3f}x")
    print(f"  Average first audio: {avg_first_audio:.2f}s")
    print(f"  Total audio        : {total_audio:.2f}s")
    print(f"  Total elapsed      : {total_elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlx_audio_repo",
        type=Path,
        default=None,
        help="Optional local mlx-audio checkout to prepend to PYTHONPATH.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="MLX Qwen3-TTS model path or repo id.",
    )
    parser.add_argument(
        "--prompts_path",
        type=Path,
        default=_default_prompts_path(),
        help="Prompt set for warmed sequential benchmarking.",
    )
    parser.add_argument(
        "--ref_audio",
        type=Path,
        default=_default_ref_audio(),
        help="Reference audio used for base-model ICL prompting.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=DEFAULT_REF_TEXT,
        help="Transcript for the reference audio.",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "cached_session", "both"),
        default="both",
        help="Which MLX path to benchmark.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable the streaming decoder path.",
    )
    parser.add_argument(
        "--streaming_interval",
        type=float,
        default=4.0,
        help="Streaming interval in seconds when --stream is enabled.",
    )
    parser.add_argument(
        "--streaming_context_size",
        type=int,
        default=25,
        help="Streaming left context size for the cached session path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed for mx.random; each prompt offsets this by its index.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum codec steps to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.5,
        help="Repetition penalty for ICL generation.",
    )
    parser.add_argument(
        "--warmup_text",
        type=str,
        default="Hi.",
        help="Warmup prompt run once after model load.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = _load_prompts(args.prompts_path)

    print(f"Prompts        : {args.prompts_path}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Stream         : {args.stream}")
    print(f"Mode           : {args.mode}")
    print()

    load_t0 = time.perf_counter()
    backend = Qwen3TTSMlxBackend(
        model_path=args.model,
        mlx_audio_repo=args.mlx_audio_repo,
    )
    load_s = time.perf_counter() - load_t0
    print(f"Device         : {backend.mx.default_device()}")
    print(f"Model load     : {load_s:.2f}s")
    if backend.repo_path is not None:
        print(f"mlx-audio repo : {backend.repo_path}")
    print()

    print("Warmup baseline generate...")
    warmup_baseline = backend.warmup(
        text=args.warmup_text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        stream=args.stream,
        streaming_interval=args.streaming_interval,
        seed=args.seed,
    )
    print(
        f"Warmup baseline: elapsed={warmup_baseline.elapsed_s:.2f}s "
        f"audio={warmup_baseline.audio_s:.2f}s "
        f"throughput={warmup_baseline.throughput_x:.3f}x"
    )

    session = backend.create_icl_session(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )
    warmup_cached = session.benchmark(
        text=args.warmup_text,
        stream=args.stream,
        streaming_interval=args.streaming_interval,
        streaming_context_size=args.streaming_context_size,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    print(
        f"Warmup cached  : elapsed={warmup_cached.elapsed_s:.2f}s "
        f"audio={warmup_cached.audio_s:.2f}s "
        f"throughput={warmup_cached.throughput_x:.3f}x"
    )

    baseline_metrics = []
    cached_metrics = []
    print()

    for prompt_idx, prompt in enumerate(prompts):
        prompt_seed = args.seed + prompt_idx
        print(f"Prompt {prompt_idx}: {prompt}")
        if args.mode in ("baseline", "both"):
            baseline = backend.benchmark_baseline(
                text=prompt,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                stream=args.stream,
                streaming_interval=args.streaming_interval,
                seed=prompt_seed,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_tokens,
            )
            baseline_metrics.append(baseline)
            print(
                "  baseline      "
                f"elapsed={baseline.elapsed_s:.2f}s "
                f"audio={baseline.audio_s:.2f}s "
                f"throughput={baseline.throughput_x:.3f}x "
                f"first_audio={baseline.first_audio_s:.2f}s "
                f"chunks={baseline.chunk_count}"
            )
        if args.mode in ("cached_session", "both"):
            cached = session.benchmark(
                text=prompt,
                stream=args.stream,
                streaming_interval=args.streaming_interval,
                streaming_context_size=args.streaming_context_size,
                seed=prompt_seed,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_tokens,
            )
            cached_metrics.append(cached)
            print(
                "  cached_session "
                f"elapsed={cached.elapsed_s:.2f}s "
                f"audio={cached.audio_s:.2f}s "
                f"throughput={cached.throughput_x:.3f}x "
                f"first_audio={cached.first_audio_s:.2f}s "
                f"chunks={cached.chunk_count}"
            )

    if baseline_metrics:
        _print_summary("Baseline mlx-audio", baseline_metrics)
    if cached_metrics:
        _print_summary("Cached session backend", cached_metrics)
    if baseline_metrics and cached_metrics:
        baseline_avg = mean(metric.throughput_x for metric in baseline_metrics)
        cached_avg = mean(metric.throughput_x for metric in cached_metrics)
        speedup = cached_avg / baseline_avg if baseline_avg > 0.0 else 0.0
        print()
        print(
            "Cached session speedup: "
            f"{speedup:.3f}x over baseline mlx-audio"
        )

    print(f"Peak memory    : {backend.mx.get_peak_memory() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
