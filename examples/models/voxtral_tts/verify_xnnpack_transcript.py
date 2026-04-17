#!/usr/bin/env python3
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from executorch.examples.models.voxtral_tts.export_voxtral_tts import (
    resolve_effective_quantization,
)
from executorch.examples.models.voxtral_tts.parity import (
    build_reference_prompt_ids,
    encode_speech_request_tokens,
)
from executorch.examples.models.voxtral_tts.voice import load_voice_from_model_dir


DEFAULT_ACCEPTANCE_ARTIFACT_DIR = "/tmp/voxtral_tts_acceptance"
DEFAULT_ACCEPTANCE_TEXT = "Hello, how are you today?"
DEFAULT_ACCEPTANCE_VOICE = "neutral_female"
DEFAULT_ACCEPTANCE_SEED = 42
DEFAULT_ACCEPTANCE_QLINEAR = "8da4w"
DEFAULT_ACCEPTANCE_DECODER_QLINEAR_SCOPE = "feed_forward"
DEFAULT_MIN_SIMILARITY = 1.0


def normalize_text(text: str) -> str:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return " ".join(tokens)


def similarity_score(expected: str, actual: str) -> float:
    return difflib.SequenceMatcher(
        None,
        normalize_text(expected),
        normalize_text(actual),
    ).ratio()


def tokenize_text(tokenizer_path: str | Path, text: str) -> list[int]:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
    inner = tokenizer.instruct_tokenizer.tokenizer
    return inner.encode(text, bos=False, eos=False)


def build_artifact_layout(artifact_dir: str | Path) -> dict[str, Path]:
    artifact_root = Path(artifact_dir)
    return {
        "artifact_dir": artifact_root,
        "export_dir": artifact_root / "export",
        "output_wav": artifact_root / "accepted.wav",
        "trace_json": artifact_root / "runner_trace.json",
        "codec_validation_json": artifact_root / "codec_validation.json",
        "stt_json": artifact_root / "apple_stt.json",
        "manifest_json": artifact_root / "manifest.json",
    }


def build_acceptance_contract(
    model_dir: str | Path,
    tokenizer_path: str | Path,
    text: str,
    voice: str | None,
    *,
    dim: int = 3072,
    begin_audio_token_id: int = 25,
    audio_token_id: int = 24,
    text_to_audio_token_id: int = 36,
    repeat_audio_text_token_id: int = 35,
) -> dict[str, Any]:
    voice_embed, voice_path = load_voice_from_model_dir(model_dir, voice, dim=dim)
    voice_name = Path(voice_path).stem
    text_tokens = tokenize_text(tokenizer_path, text)
    prompt = build_reference_prompt_ids(
        text_tokens=text_tokens,
        voice_len=int(voice_embed.shape[0]),
        begin_audio_token_id=begin_audio_token_id,
        audio_token_id=audio_token_id,
        text_to_audio_token_id=text_to_audio_token_id,
        repeat_audio_text_token_id=repeat_audio_text_token_id,
    )
    official_prompt_ids = encode_speech_request_tokens(tokenizer_path, text, voice_name)
    if prompt.token_ids != official_prompt_ids:
        raise RuntimeError(
            "Manual prompt construction diverges from mistral_common "
            f"encode_speech_request for voice={voice_name}"
        )
    return {
        "text": text,
        "normalized_text": normalize_text(text),
        "voice_name": voice_name,
        "voice_path": str(voice_path),
        "voice_len": int(voice_embed.shape[0]),
        "voice_start": prompt.voice_start,
        "prompt_token_ids": official_prompt_ids,
    }


def evaluate_transcript_gate(
    expected: str,
    actual: str,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> dict[str, Any]:
    normalized_expected = normalize_text(expected)
    normalized_actual = normalize_text(actual)
    score = similarity_score(expected, actual)
    if not normalized_actual:
        return {
            "ok": False,
            "reason": "empty_transcript",
            "score": score,
            "normalized_expected": normalized_expected,
            "normalized_actual": normalized_actual,
        }
    if normalized_actual == "no speech detected":
        return {
            "ok": False,
            "reason": "no_speech_detected",
            "score": score,
            "normalized_expected": normalized_expected,
            "normalized_actual": normalized_actual,
        }
    if normalized_actual != normalized_expected and score < min_similarity:
        return {
            "ok": False,
            "reason": "normalized_text_mismatch",
            "score": score,
            "normalized_expected": normalized_expected,
            "normalized_actual": normalized_actual,
        }
    return {
        "ok": True,
        "reason": "match",
        "score": score,
        "normalized_expected": normalized_expected,
        "normalized_actual": normalized_actual,
    }


def build_export_command(
    repo_root: str | Path,
    *,
    model_dir: str | Path,
    export_dir: str | Path,
    max_seq_len: int,
    max_codec_frames: int,
    qlinear: str | None,
    qembedding: str | None,
    decoder_qlinear_scope: str,
) -> list[str]:
    repo_root = Path(repo_root)
    export_script = repo_root / "examples/models/voxtral_tts/export_voxtral_tts.py"
    command = [
        sys.executable,
        str(export_script),
        "--model-path",
        str(model_dir),
        "--backend",
        "xnnpack",
        "--max-seq-len",
        str(max_seq_len),
        "--max-codec-frames",
        str(max_codec_frames),
        "--output-dir",
        str(export_dir),
    ]
    if qlinear is not None and qlinear != "none":
        command.extend(["--qlinear", qlinear])
        command.extend(["--decoder-qlinear-scope", decoder_qlinear_scope])
    if qembedding is not None and qembedding != "none":
        command.extend(["--qembedding", qembedding])
    return command


def build_runner_command(
    *,
    repo_root: str | Path,
    layout: dict[str, Path],
    tokenizer_path: str | Path,
    voice_path: str | Path,
    text: str,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    repo_root = Path(repo_root)
    runner = repo_root / "cmake-out/examples/models/voxtral_tts/voxtral_tts_runner"
    return [
        str(runner),
        "--model",
        str(layout["export_dir"] / "model.pte"),
        "--codec",
        str(layout["export_dir"] / "codec_decoder.pte"),
        "--tokenizer",
        str(tokenizer_path),
        "--voice",
        str(voice_path),
        "--text",
        text,
        "--output",
        str(layout["output_wav"]),
        "--trace_json",
        str(layout["trace_json"]),
        "--max_new_tokens",
        str(max_new_tokens),
        "--seed",
        str(seed),
    ]


def build_stt_command(
    repo_root: str | Path,
    *,
    output_wav: str | Path,
    locale: str,
) -> list[str]:
    """Build STT command using parakeet runner (cross-platform, replaces Apple STT).

    The parakeet runner expects 16 kHz input. This function returns a shell
    command that resamples the 24 kHz Voxtral WAV to 16 kHz and transcribes it.
    """
    repo_root = Path(repo_root)
    parakeet_runner = (
        repo_root / "cmake-out/examples/models/parakeet/parakeet_runner"
    )
    parakeet_model = (
        repo_root / "examples/models/parakeet/parakeet_tdt_exports/model.pte"
    )
    parakeet_tokenizer = (
        repo_root / "examples/models/parakeet/parakeet_tdt_exports/tokenizer.model"
    )
    # We use a helper Python script to resample + run + extract transcript.
    resample_and_transcribe = repo_root / "examples/models/voxtral_tts/transcribe_parakeet.py"
    return [
        sys.executable,
        str(resample_and_transcribe),
        "--audio", str(output_wav),
        "--parakeet-runner", str(parakeet_runner),
        "--parakeet-model", str(parakeet_model),
        "--parakeet-tokenizer", str(parakeet_tokenizer),
    ]


def build_codec_validation_command(
    repo_root: str | Path,
    *,
    model_dir: str | Path,
    layout: dict[str, Path],
    max_seq_len: int,
    max_codec_frames: int,
) -> list[str]:
    repo_root = Path(repo_root)
    codec_script = repo_root / "examples/models/voxtral_tts/verify_codec_export.py"
    return [
        sys.executable,
        str(codec_script),
        "--model-path",
        str(model_dir),
        "--codec-pte",
        str(layout["export_dir"] / "codec_decoder.pte"),
        "--trace-json",
        str(layout["trace_json"]),
        "--max-seq-len",
        str(max_seq_len),
        "--max-codec-frames",
        str(max_codec_frames),
        "--output-json",
        str(layout["codec_validation_json"]),
    ]


def build_acceptance_manifest(
    *,
    layout: dict[str, Path],
    contract: dict[str, Any],
    export_args: dict[str, Any],
    runner_args: dict[str, Any],
    codec_validation: dict[str, Any] | None,
    transcript: str | None,
    transcript_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "artifact_dir": str(layout["artifact_dir"]),
        "paths": {
            "export_dir": str(layout["export_dir"]),
            "output_wav": str(layout["output_wav"]),
            "trace_json": str(layout["trace_json"]),
            "codec_validation_json": str(layout["codec_validation_json"]),
            "stt_json": str(layout["stt_json"]),
            "manifest_json": str(layout["manifest_json"]),
        },
        "contract": contract,
        "export_args": export_args,
        "runner_args": runner_args,
        "codec_validation": codec_validation,
        "transcript": transcript,
        "transcript_gate": transcript_gate,
        "ok": bool(
            codec_validation
            and codec_validation["ok"]
            and transcript_gate
            and transcript_gate["ok"]
        ),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def run_checked(
    command: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export Voxtral TTS for XNNPACK, generate a WAV, and hard-fail on "
            "STT transcript mismatch (uses parakeet runner)."
        )
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--artifact-dir", default=DEFAULT_ACCEPTANCE_ARTIFACT_DIR)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--output-wav", default=None)
    parser.add_argument("--voice", default=DEFAULT_ACCEPTANCE_VOICE)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", default=DEFAULT_ACCEPTANCE_TEXT)
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--seed", type=int, default=DEFAULT_ACCEPTANCE_SEED)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-codec-frames", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--qlinear", default=DEFAULT_ACCEPTANCE_QLINEAR,
                        help="Quantization config, or 'none' for FP32.")
    parser.add_argument(
        "--decoder-qlinear-scope",
        default=DEFAULT_ACCEPTANCE_DECODER_QLINEAR_SCOPE,
        choices=["all", "attention", "feed_forward", "none"],
    )
    parser.add_argument("--qembedding", default=None, choices=["4w", "8w"])
    args = parser.parse_args()
    quant_plan = resolve_effective_quantization(
        backend="xnnpack",
        qlinear=args.qlinear,
        qembedding=args.qembedding,
    )
    effective_qlinear = quant_plan["qlinear"]
    effective_qembedding = quant_plan["qembedding"]

    repo_root = Path(args.repo_root).resolve()
    layout = build_artifact_layout(args.artifact_dir)
    if args.export_dir:
        layout["export_dir"] = Path(args.export_dir)
    if args.output_wav:
        layout["output_wav"] = Path(args.output_wav)

    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        env["PATH"] = f"{conda_prefix}/bin:{env.get('PATH', '')}"

    layout["artifact_dir"].mkdir(parents=True, exist_ok=True)
    layout["export_dir"].mkdir(parents=True, exist_ok=True)

    contract = build_acceptance_contract(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer,
        text=args.text,
        voice=args.voice,
    )

    export_args = {
        "backend": "xnnpack",
        "model_dir": str(args.model_dir),
        "max_seq_len": args.max_seq_len,
        "max_codec_frames": args.max_codec_frames,
        "qlinear": effective_qlinear,
        "qembedding": effective_qembedding,
        "decoder_qlinear_scope": args.decoder_qlinear_scope,
        "requested_qlinear": args.qlinear,
        "requested_qembedding": args.qembedding,
        "quantization_warning": quant_plan["warning"],
    }
    runner_args = {
        "tokenizer": args.tokenizer,
        "voice_path": contract["voice_path"],
        "text": args.text,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }

    manifest = build_acceptance_manifest(
        layout=layout,
        contract=contract,
        export_args=export_args,
        runner_args=runner_args,
        codec_validation=None,
        transcript=None,
        transcript_gate=None,
    )
    write_json(layout["manifest_json"], manifest)

    try:
        run_checked(
            build_export_command(
                repo_root,
                model_dir=args.model_dir,
                export_dir=layout["export_dir"],
                max_seq_len=args.max_seq_len,
                max_codec_frames=args.max_codec_frames,
                qlinear=effective_qlinear,
                qembedding=effective_qembedding,
                decoder_qlinear_scope=args.decoder_qlinear_scope,
            ),
            env=env,
        )
        run_checked(
            build_runner_command(
                repo_root=repo_root,
                layout=layout,
                tokenizer_path=args.tokenizer,
                voice_path=contract["voice_path"],
                text=args.text,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            ),
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        elif exc.stdout:
            print(exc.stdout, file=sys.stderr, end="")
        return 1

    codec_validation = None
    try:
        run_checked(
            build_codec_validation_command(
                repo_root,
                model_dir=args.model_dir,
                layout=layout,
                max_seq_len=args.max_seq_len,
                max_codec_frames=args.max_codec_frames,
            ),
            env=env,
        )
        codec_validation = read_json(layout["codec_validation_json"])
    except subprocess.CalledProcessError as exc:
        if layout["codec_validation_json"].exists():
            codec_validation = read_json(layout["codec_validation_json"])
        manifest = build_acceptance_manifest(
            layout=layout,
            contract=contract,
            export_args=export_args,
            runner_args=runner_args,
            codec_validation=codec_validation,
            transcript=None,
            transcript_gate=None,
        )
        write_json(layout["manifest_json"], manifest)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        elif exc.stdout:
            print(exc.stdout, file=sys.stderr, end="")
        return 1

    manifest = build_acceptance_manifest(
        layout=layout,
        contract=contract,
        export_args=export_args,
        runner_args=runner_args,
        codec_validation=codec_validation,
        transcript=None,
        transcript_gate=None,
    )
    write_json(layout["manifest_json"], manifest)
    if not codec_validation["ok"]:
        print(
            f"Codec validation failed: max_abs_diff={codec_validation['max_abs_diff']:.6f}",
            file=sys.stderr,
        )
        return 1

    try:
        transcript_result = run_checked(
            build_stt_command(
                repo_root,
                output_wav=layout["output_wav"],
                locale=args.locale,
            ),
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        elif exc.stdout:
            print(exc.stdout, file=sys.stderr, end="")
        return 1

    transcript = transcript_result.stdout.strip()
    transcript_gate = evaluate_transcript_gate(
        args.text,
        transcript,
        min_similarity=args.min_similarity,
    )
    write_json(
        layout["stt_json"],
        {
            "locale": args.locale,
            "transcript": transcript,
            **transcript_gate,
        },
    )

    manifest = build_acceptance_manifest(
        layout=layout,
        contract=contract,
        export_args=export_args,
        runner_args=runner_args,
        codec_validation=codec_validation,
        transcript=transcript,
        transcript_gate=transcript_gate,
    )
    write_json(layout["manifest_json"], manifest)

    if not transcript_gate["ok"]:
        print(
            f"STT gate failed: {transcript_gate['reason']} "
            f"(score={transcript_gate['score']:.6f})",
            file=sys.stderr,
        )
        return 1

    print(f"{transcript_gate['score']:.6f}")
    print(f"TRANSCRIPT: {transcript}", file=sys.stderr)
    print(f"MANIFEST: {layout['manifest_json']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
