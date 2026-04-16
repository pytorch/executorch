from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def _load_codec_module():
    module_path = Path(__file__).resolve().with_name("verify_codec_export.py")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("voxtral_verify_codec_export", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decode_exported_waveform_falls_back_to_padded_window() -> None:
    module = _load_codec_module()

    calls: list[tuple[int, int]] = []

    class FakeExported:
        def forward(self, inputs):
            (codes,) = inputs
            frames = int(codes.shape[2])
            calls.append((frames, int(codes[0, 0, 0].item())))
            if frames == 3:
                raise RuntimeError("expected fixed codec window")
            return (
                torch.arange(12, dtype=torch.float32).view(1, 1, 12),
            )

    codes = torch.tensor([[[7, 8, 9]]], dtype=torch.long)

    waveform, mode = module.decode_exported_waveform(
        FakeExported(),
        codes,
        valid_samples=6,
        max_codec_frames=6,
    )

    assert mode == "padded"
    assert calls == [(3, 7), (6, 7)]
    assert waveform.shape == (1, 1, 6)
    assert waveform.tolist() == [[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]


def test_decode_exported_waveform_raises_without_padding_budget() -> None:
    module = _load_codec_module()

    class FakeExported:
        def forward(self, inputs):
            raise RuntimeError("expected fixed codec window")

    codes = torch.tensor([[[1, 2, 3]]], dtype=torch.long)

    with pytest.raises(RuntimeError, match="expected fixed codec window"):
        module.decode_exported_waveform(
            FakeExported(),
            codes,
            valid_samples=6,
            max_codec_frames=None,
        )


def test_decode_reference_waveform_uses_padded_mode_and_trims() -> None:
    module = _load_codec_module()

    calls: list[int] = []

    class FakeCodec:
        def __call__(self, codes):
            calls.append(int(codes.shape[2]))
            return torch.arange(12, dtype=torch.float32).view(1, 1, 12)

    codes = torch.tensor([[[7, 8, 9]]], dtype=torch.long)

    waveform = module.decode_reference_waveform(
        FakeCodec(),
        codes,
        mode="padded",
        valid_samples=6,
        max_codec_frames=6,
    )

    assert calls == [6]
    assert waveform.shape == (1, 1, 6)
    assert waveform.tolist() == [[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]
