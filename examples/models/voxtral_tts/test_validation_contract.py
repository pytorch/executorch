from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


def _load_validation_module():
    module_path = Path(__file__).resolve().with_name("verify_xnnpack_transcript.py")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("voxtral_tts_validation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_artifact_layout_uses_single_bundle_root(tmp_path: Path) -> None:
    module = _load_validation_module()

    layout = module.build_artifact_layout(tmp_path)

    assert layout["artifact_dir"] == tmp_path
    assert layout["export_dir"] == tmp_path / "export"
    assert layout["output_wav"] == tmp_path / "accepted.wav"
    assert layout["trace_json"] == tmp_path / "runner_trace.json"
    assert layout["codec_validation_json"] == tmp_path / "codec_validation.json"
    assert layout["stt_json"] == tmp_path / "apple_stt.json"
    assert layout["manifest_json"] == tmp_path / "manifest.json"


def test_build_acceptance_contract_resolves_voice_and_prompt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_validation_module()
    voice_path = tmp_path / "voice_embedding" / "neutral_female.pt"
    voice_path.parent.mkdir()

    monkeypatch.setattr(
        module,
        "load_voice_from_model_dir",
        lambda model_dir, voice, dim=3072: (torch.zeros(3, dim), voice_path),
    )
    monkeypatch.setattr(module, "tokenize_text", lambda tokenizer_path, text: [101, 102])
    monkeypatch.setattr(
        module,
        "encode_speech_request_tokens",
        lambda tokenizer_path, text, voice_name: [1, 25, 24, 24, 24, 36, 101, 102, 35, 25],
    )

    contract = module.build_acceptance_contract(
        model_dir=tmp_path,
        tokenizer_path=tmp_path / "tekken.json",
        text="Hello world",
        voice=None,
    )

    assert contract["text"] == "Hello world"
    assert contract["voice_name"] == "neutral_female"
    assert contract["voice_path"] == str(voice_path)
    assert contract["voice_len"] == 3
    assert contract["voice_start"] == 2
    assert contract["prompt_token_ids"] == [1, 25, 24, 24, 24, 36, 101, 102, 35, 25]


def test_evaluate_transcript_gate_rejects_no_speech_and_requires_match() -> None:
    module = _load_validation_module()

    ok = module.evaluate_transcript_gate("Hello, world!", "hello world")
    assert ok["ok"] is True
    assert ok["score"] == 1.0

    no_speech = module.evaluate_transcript_gate("Hello, world!", "No speech detected")
    assert no_speech["ok"] is False
    assert no_speech["reason"] == "no_speech_detected"

    mismatch = module.evaluate_transcript_gate("Hello, world!", "hello there")
    assert mismatch["ok"] is False
    assert mismatch["reason"] == "normalized_text_mismatch"


def test_build_runner_command_threads_seed_trace_and_resolved_voice(
    tmp_path: Path,
) -> None:
    module = _load_validation_module()
    layout = module.build_artifact_layout(tmp_path)

    command = module.build_runner_command(
        repo_root=tmp_path,
        layout=layout,
        tokenizer_path=tmp_path / "tekken.json",
        voice_path=tmp_path / "voice_embedding" / "neutral_female.pt",
        text="Hello world",
        max_new_tokens=24,
        seed=17,
    )

    assert command[:1] == [str(tmp_path / "cmake-out/examples/models/voxtral_tts/voxtral_tts_runner")]
    assert "--trace_json" in command
    assert str(layout["trace_json"]) in command
    assert "--seed" in command
    assert "17" in command
    assert "--voice" in command
    assert str(tmp_path / "voice_embedding" / "neutral_female.pt") in command


def test_build_export_command_threads_decoder_qlinear_scope(
    tmp_path: Path,
) -> None:
    module = _load_validation_module()

    command = module.build_export_command(
        tmp_path,
        model_dir=tmp_path / "model_dir",
        export_dir=tmp_path / "export",
        max_seq_len=512,
        max_codec_frames=64,
        qlinear="8da8w",
        qembedding=None,
        decoder_qlinear_scope="feed_forward",
    )

    assert command[:2] == [
        sys.executable,
        str(tmp_path / "examples/models/voxtral_tts/export_voxtral_tts.py"),
    ]
    assert "--qlinear" in command
    assert "8da8w" in command
    assert "--decoder-qlinear-scope" in command
    assert "feed_forward" in command


def test_build_codec_validation_command_uses_runner_trace_bundle(
    tmp_path: Path,
) -> None:
    module = _load_validation_module()
    layout = module.build_artifact_layout(tmp_path)

    command = module.build_codec_validation_command(
        repo_root=tmp_path,
        model_dir=tmp_path / "model_dir",
        layout=layout,
        max_seq_len=512,
        max_codec_frames=64,
    )

    assert command[:2] == [
        sys.executable,
        str(tmp_path / "examples/models/voxtral_tts/verify_codec_export.py"),
    ]
    assert "--codec-pte" in command
    assert str(layout["export_dir"] / "codec_decoder.pte") in command
    assert "--trace-json" in command
    assert str(layout["trace_json"]) in command
    assert "--output-json" in command
    assert str(layout["codec_validation_json"]) in command
    assert "--max-codec-frames" in command
    assert "64" in command
