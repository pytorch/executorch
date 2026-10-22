# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch

from examples.models.supertonic.model.config import TTSConfig


def _duration_predictor():
    return importlib.import_module(
        "examples.models.supertonic.model.duration_predictor"
    )


def _config() -> TTSConfig:
    return TTSConfig.from_dict(
        {
            "tts_version": "test",
            "split": "test",
            "ttl": {"latent_dim": 4, "chunk_compress_factor": 2},
            "ae": {
                "sample_rate": 16000,
                "base_chunk_size": 4,
                "chunk_compress_factor": 1,
                "ldim": 4,
            },
            "dp": {"latent_dim": 4, "chunk_compress_factor": 2},
        }
    )


def _small_model():
    return (
        _duration_predictor()
        .DurationPredictor(
            _config(),
            vocab_size=32,
            channels=8,
            convnext_dilations=(1, 1),
            attention_layers=1,
            attention_heads=2,
            ff_channels=16,
            relative_window=2,
            style_tokens=2,
            style_dim=3,
            hidden_dim=10,
        )
        .eval()
    )


def _contract_model():
    return (
        _duration_predictor()
        .DurationPredictor(
            _config(),
            vocab_size=8,
            channels=4,
            convnext_dilations=(),
            attention_layers=0,
            attention_heads=1,
            ff_channels=4,
            style_tokens=8,
            style_dim=16,
            hidden_dim=4,
        )
        .eval()
    )


def test_duration_predictor_returns_one_finite_value_per_batch() -> None:
    torch.manual_seed(0)
    model = _small_model()

    output = model(
        torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        torch.randn(2, 2, 3),
        torch.tensor([[[1.0, 1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0, 0.0]]]),
    )

    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert (output > 0).all()


def test_duration_predictor_ignores_masked_text_and_is_deterministic() -> None:
    torch.manual_seed(1)
    model = _small_model()
    text_ids = torch.tensor([[1, 2, 3, 4]])
    changed_ids = torch.tensor([[1, 2, 30, 31]])
    style = torch.randn(1, 2, 3)
    mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])

    first = model(text_ids, style, mask)
    second = model(changed_ids, style, mask)
    repeated = model(text_ids, style, mask)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, repeated)


@pytest.mark.parametrize(
    ("text_shape", "style_shape", "mask_shape", "error"),
    [
        ((3,), (1, 8, 16), (1, 1, 3), r"text_ids.*\[B, T\]"),
        ((1, 3), (1, 8, 15), (1, 1, 3), r"style_dp.*\[B, 8, 16\]"),
        ((1, 3), (1, 8, 16), (1, 2, 3), r"text_mask.*\[B, 1, T\]"),
        ((1, 3), (2, 8, 16), (1, 1, 3), "batch sizes must match"),
        ((1, 3), (1, 8, 16), (1, 1, 2), "text lengths must match"),
    ],
)
def test_duration_predictor_validates_public_input_contract_before_operators(
    text_shape, style_shape, mask_shape, error: str
) -> None:
    model = _contract_model()
    text_ids = torch.full(text_shape, 999, dtype=torch.long)
    style = torch.zeros(style_shape)
    mask = torch.ones(mask_shape)

    with pytest.raises(ValueError, match=error):
        model(text_ids, style, mask)
