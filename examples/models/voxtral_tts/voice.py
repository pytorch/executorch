from pathlib import Path

import numpy as np
import torch


DEFAULT_VOICE_NAME = "neutral_female"


def resolve_voice_asset_path(model_dir: str | Path, voice: str | None) -> Path:
    model_dir = Path(model_dir)
    voice_name = voice or DEFAULT_VOICE_NAME
    candidate = Path(voice_name)

    if candidate.exists():
        return candidate

    voice_dir = model_dir / "voice_embedding"
    if candidate.suffix:
        local_candidate = voice_dir / candidate.name
        if local_candidate.exists():
            return local_candidate
        return candidate

    for ext in (".pt", ".bin"):
        local_candidate = voice_dir / f"{voice_name}{ext}"
        if local_candidate.exists():
            return local_candidate

    return voice_dir / f"{voice_name}.pt"


def load_voice_embedding_tensor(
    path: str | Path,
    dim: int = 3072,
    expected_frames_hint: int | None = None,
) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True).float()

    raw = path.read_bytes()
    bf16_row_bytes = dim * 2
    f32_row_bytes = dim * 4
    matches_hint_bf16 = (
        expected_frames_hint is not None
        and len(raw) == expected_frames_hint * bf16_row_bytes
    )
    matches_hint_f32 = (
        expected_frames_hint is not None
        and len(raw) == expected_frames_hint * f32_row_bytes
    )

    if matches_hint_f32:
        data = np.frombuffer(raw, dtype=np.float32).copy()
        return torch.from_numpy(data).reshape(-1, dim).float()

    if matches_hint_bf16 or len(raw) % bf16_row_bytes == 0:
        data = np.frombuffer(raw, dtype=np.uint16).copy()
        tensor = torch.from_numpy(data).reshape(-1, dim)
        return tensor.view(torch.bfloat16).float()

    if len(raw) % f32_row_bytes == 0:
        data = np.frombuffer(raw, dtype=np.float32).copy()
        return torch.from_numpy(data).reshape(-1, dim).float()

    raise ValueError(
        f"Voice embedding {path} has unsupported size {len(raw)} for dim={dim}"
    )


def load_voice_from_model_dir(
    model_dir: str | Path,
    voice: str | None,
    dim: int = 3072,
) -> tuple[torch.Tensor, Path]:
    path = resolve_voice_asset_path(model_dir, voice)
    expected_frames_hint = None
    if path.suffix == ".bin":
        pt_peer = path.with_suffix(".pt")
        if pt_peer.exists():
            expected_frames_hint = int(
                load_voice_embedding_tensor(pt_peer, dim=dim).shape[0]
            )
    return (
        load_voice_embedding_tensor(
            path,
            dim=dim,
            expected_frames_hint=expected_frames_hint,
        ),
        path,
    )
