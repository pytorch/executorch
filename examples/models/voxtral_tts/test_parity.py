from pathlib import Path

import torch

from executorch.examples.models.voxtral_tts.parity import (
    build_reference_prompt_ids,
    compare_trace_payloads,
    run_seed_decode,
)
from executorch.examples.models.voxtral_tts.voice import (
    DEFAULT_VOICE_NAME,
    load_voice_embedding_tensor,
    load_voice_from_model_dir,
    resolve_voice_asset_path,
)


class DummyTokenEmbedding(torch.nn.Module):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.to(torch.float32).unsqueeze(-1)


class RecordingDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self, input_embeds: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((input_embeds.clone(), positions.clone()))
        if input_embeds.shape[1] > 1:
            return positions.to(torch.float32).view(1, -1, 1) + 100.0
        return positions.to(torch.float32).view(1, 1, 1) + 200.0


def test_build_reference_prompt_omits_audio_placeholders_without_voice():
    prompt = build_reference_prompt_ids(
        text_tokens=[101, 102],
        voice_len=0,
        begin_audio_token_id=25,
        audio_token_id=24,
        text_to_audio_token_id=36,
        repeat_audio_text_token_id=35,
    )

    assert prompt.token_ids == [1, 25, 36, 101, 102, 35, 25]
    assert prompt.voice_start == 2
    assert prompt.voice_len == 0


def test_build_reference_prompt_uses_runtime_voice_length():
    prompt = build_reference_prompt_ids(
        text_tokens=[101],
        voice_len=3,
        begin_audio_token_id=25,
        audio_token_id=24,
        text_to_audio_token_id=36,
        repeat_audio_text_token_id=35,
    )

    assert prompt.token_ids == [1, 25, 24, 24, 24, 36, 101, 35, 25]
    assert prompt.voice_start == 2
    assert prompt.voice_len == 3


def test_run_seed_decode_feeds_explicit_audio_token_after_prefill():
    token_embedding = DummyTokenEmbedding()
    decoder = RecordingDecoder()
    prompt_embeds = torch.zeros(1, 4, 1)

    trace = run_seed_decode(
        token_embedding=token_embedding,
        decoder=decoder,
        audio_token_id=24,
        prompt_embeds=prompt_embeds,
    )

    assert trace.prefill_hidden.squeeze().item() == 103.0
    assert trace.seed_hidden.squeeze().item() == 204.0
    assert trace.seed_position == 4

    assert len(decoder.calls) == 2
    seed_input_embeds, seed_positions = decoder.calls[1]
    assert seed_positions.tolist() == [4]
    assert seed_input_embeds.shape == (1, 1, 1)
    assert seed_input_embeds.squeeze().item() == 24.0


def test_compare_trace_payloads_flags_hidden_and_code_mismatches():
    reference = {
        "prompt_token_ids": [1, 25, 24, 36, 101, 35, 25],
        "voice_len": 1,
        "prefill_hidden": [0.0, 1.0],
        "frame0_hidden": [2.0, 3.0],
        "seed_hidden": [2.0, 3.0],
        "seed_position": 7,
        "frame0_position": 7,
        "frame0_full_codes": [7, 10, 11],
        "frame0_audio_embed": [0.5, -0.5],
        "frame1_position": 8,
        "frame1_hidden": [4.0, 5.0],
        "frames": [
            {
                "semantic_code": 7,
                "full_codes": [7, 10, 11],
            }
        ],
    }
    candidate = {
        "prompt_token_ids": [1, 25, 24, 36, 101, 35, 25],
        "voice_len": 1,
        "prefill_hidden": [0.0, 1.0],
        "frame0_hidden": [2.5, 3.0],
        "seed_hidden": [2.5, 3.0],
        "seed_position": 7,
        "frame0_position": 7,
        "frame0_full_codes": [8, 10, 11],
        "frame0_audio_embed": [0.75, -0.5],
        "frame1_position": 8,
        "frame1_hidden": [4.5, 5.0],
        "frames": [
            {
                "semantic_code": 8,
                "full_codes": [8, 10, 11],
            }
        ],
    }

    result = compare_trace_payloads(reference, candidate, hidden_atol=1e-4)

    assert result["ok"] is False
    failed_names = {check["name"] for check in result["checks"] if not check["ok"]}
    assert "frame0_hidden" in failed_names
    assert "seed_hidden" in failed_names
    assert "frame0_semantic_code" in failed_names
    assert "frame0_full_codes" in failed_names
    assert "frame0_audio_embed" in failed_names
    assert "frame0_codes" in failed_names
    assert "frame1_hidden" in failed_names


def test_resolve_voice_asset_path_defaults_to_neutral_female_pt(tmp_path: Path):
    voice_dir = tmp_path / "voice_embedding"
    voice_dir.mkdir()
    target = voice_dir / f"{DEFAULT_VOICE_NAME}.pt"
    target.write_bytes(b"stub")

    assert resolve_voice_asset_path(tmp_path, None) == target


def test_resolve_voice_asset_path_falls_back_to_bin_for_named_voice(tmp_path: Path):
    voice_dir = tmp_path / "voice_embedding"
    voice_dir.mkdir()
    target = voice_dir / "casual_male.bin"
    target.write_bytes(b"stub")

    assert resolve_voice_asset_path(tmp_path, "casual_male") == target


def test_load_voice_embedding_tensor_reads_pt_and_bin(tmp_path: Path):
    expected = torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.bfloat16)

    pt_path = tmp_path / "voice.pt"
    torch.save(expected, pt_path)
    loaded_pt = load_voice_embedding_tensor(pt_path, dim=2)
    assert torch.equal(loaded_pt, expected.float())

    bin_path = tmp_path / "voice.bin"
    bin_path.write_bytes(expected.view(torch.int16).numpy().tobytes())
    loaded_bin = load_voice_embedding_tensor(bin_path, dim=2)
    assert torch.equal(loaded_bin, expected.float())


def test_load_voice_from_model_dir_uses_pt_peer_to_disambiguate_float32_bin(
    tmp_path: Path,
):
    voice_dir = tmp_path / "voice_embedding"
    voice_dir.mkdir()

    expected = torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.float32)
    pt_peer = voice_dir / "casual_male.pt"
    torch.save(expected.to(torch.bfloat16), pt_peer)

    bin_path = voice_dir / "casual_male.bin"
    bin_path.write_bytes(expected.numpy().tobytes())

    loaded, resolved = load_voice_from_model_dir(tmp_path, "casual_male.bin", dim=2)
    assert resolved == bin_path
    assert torch.equal(loaded, expected)
