# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch

from examples.models.supertonic.model.config import TTSConfig


def _text_encoder():
    return importlib.import_module("examples.models.supertonic.model.text_encoder")


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
        _text_encoder()
        .TextEncoder(
            _config(),
            vocab_size=32,
            channels=8,
            convnext_dilations=(1, 2),
            attention_layers=1,
            attention_heads=2,
            ff_channels=16,
            relative_window=2,
            style_tokens=3,
            style_attention_heads=2,
        )
        .eval()
    )


def _contract_model():
    return (
        _text_encoder()
        .TextEncoder(
            _config(),
            vocab_size=8,
            channels=256,
            convnext_dilations=(),
            attention_layers=0,
            attention_heads=1,
            ff_channels=4,
            style_tokens=50,
            style_attention_heads=2,
        )
        .eval()
    )


def test_text_encoder_returns_finite_channel_first_embeddings() -> None:
    torch.manual_seed(0)
    model = _small_model()

    output = model(
        torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        torch.randn(2, 3, 8),
        torch.tensor([[[1.0, 1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0, 0.0]]]),
    )

    assert output.shape == (2, 8, 4)
    assert torch.isfinite(output).all()


def test_text_encoder_ignores_and_zeroes_masked_text_deterministically() -> None:
    torch.manual_seed(1)
    model = _small_model()
    text_ids = torch.tensor([[1, 2, 3, 4]])
    changed_ids = torch.tensor([[1, 2, 30, 31]])
    style = torch.randn(1, 3, 8)
    mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])

    first = model(text_ids, style, mask)
    second = model(changed_ids, style, mask)
    repeated = model(text_ids, style, mask)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first[:, :, 2:], torch.zeros_like(first[:, :, 2:]))


@pytest.mark.parametrize(
    ("text_shape", "style_shape", "mask_shape", "error"),
    [
        ((3,), (1, 50, 256), (1, 1, 3), r"text_ids.*\[B, T\]"),
        ((1, 3), (1, 49, 256), (1, 1, 3), r"style_ttl.*\[B, 50, 256\]"),
        ((1, 3), (1, 50, 256), (1, 2, 3), r"text_mask.*\[B, 1, T\]"),
        ((1, 3), (2, 50, 256), (1, 1, 3), "batch sizes must match"),
        ((1, 3), (1, 50, 256), (1, 1, 2), "text lengths must match"),
    ],
)
def test_text_encoder_validates_public_input_contract_before_operators(
    text_shape, style_shape, mask_shape, error: str
) -> None:
    model = _contract_model()
    text_ids = torch.full(text_shape, 999, dtype=torch.long)
    style = torch.zeros(style_shape)
    mask = torch.ones(mask_shape)

    with pytest.raises(ValueError, match=error):
        model(text_ids, style, mask)


def test_relative_position_conversions_round_trip_absolute_positions() -> None:
    attention = _text_encoder().RelativeMultiHeadAttention(
        channels=6, num_heads=2, window_size=2
    )
    absolute = torch.arange(2 * 2 * 3 * 3, dtype=torch.float32).reshape(2, 2, 3, 3)

    relative = attention._absolute_to_relative(absolute)
    restored_absolute = attention._relative_to_absolute(relative)
    restored_relative = attention._absolute_to_relative(restored_absolute)

    torch.testing.assert_close(restored_absolute, absolute)
    torch.testing.assert_close(restored_relative, relative)


def test_style_attention_splits_heads_on_leading_axis() -> None:
    attention = _text_encoder().TanhKeyAttention(channels=4, num_heads=2)
    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ]
    )

    heads = attention._split_heads(values)

    assert heads.shape == (2, 2, 1, 2)
    torch.testing.assert_close(heads[0], values[:, :, :2])
    torch.testing.assert_close(heads[1], values[:, :, 2:])
