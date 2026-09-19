# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch

from examples.models.supertonic.model.config import TTSConfig


def _vocoder():
    return importlib.import_module("examples.models.supertonic.model.vocoder")


def _config() -> TTSConfig:
    return TTSConfig.from_dict(
        {
            "tts_version": "test",
            "split": "test",
            "ttl": {"latent_dim": 2, "chunk_compress_factor": 3},
            "ae": {
                "sample_rate": 16000,
                "base_chunk_size": 4,
                "chunk_compress_factor": 1,
                "ldim": 2,
            },
            "dp": {"latent_dim": 2, "chunk_compress_factor": 3},
        }
    )


def _small_model():
    return (
        _vocoder()
        .Vocoder(
            _config(),
            decoder_channels=8,
            decoder_dilations=(1, 2),
            decoder_expansion=2,
            head_hidden_channels=12,
        )
        .eval()
    )


def test_vocoder_returns_finite_waveform_with_exact_length_deterministically() -> None:
    torch.manual_seed(0)
    model = _small_model()
    latent = torch.randn(2, 6, 5)

    first = model(latent)
    repeated = model(latent)

    assert first.shape == (2, 5 * 3 * 4)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, repeated)


def test_vocoder_unpacks_compressed_latent_in_exported_order() -> None:
    latent = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)

    unpacked = _vocoder().Vocoder._unpack_latent(
        latent,
        latent_dim=2,
        compress_factor=3,
    )
    expected = latent.reshape(1, 2, 3, 4).transpose(2, 3).reshape(1, 2, 12)

    torch.testing.assert_close(unpacked, expected)


def test_causal_padding_replicates_only_the_exported_left_context() -> None:
    pad = _vocoder().CausalPad1d(kernel_size=3, dilation=2)
    inputs = torch.tensor([[[1.0, 2.0, 3.0]]])

    output = pad(inputs)

    torch.testing.assert_close(
        output,
        torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]]]),
    )


def test_decoder_head_applies_shared_prelu_and_flattens_time_major() -> None:
    head = _vocoder().DecoderHead(
        channels=1,
        hidden_channels=1,
        output_channels=2,
    )
    with torch.no_grad():
        head.layer1.net.weight.zero_()
        head.layer1.net.weight[:, :, 2] = 1.0
        head.layer1.net.bias.zero_()
        head.act.weight.fill_(0.25)
        head.layer2.weight.copy_(torch.tensor([[[1.0]], [[-1.0]]]))

    output = head(torch.tensor([[[-1.0, 2.0]]]))

    torch.testing.assert_close(
        output,
        torch.tensor([[-0.25, 0.25, 2.0, -2.0]]),
    )


@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ((6, 4), r"latent.*\[B, 6, L\]"),
        ((1, 5, 4), r"latent.*\[B, 6, L\]"),
        ((1, 6, 0), "latent length must be positive"),
    ],
)
def test_vocoder_validates_public_input_contract_before_operators(
    shape: tuple[int, ...], error: str
) -> None:
    model = _small_model()

    with pytest.raises(ValueError, match=error):
        model(torch.zeros(shape))
