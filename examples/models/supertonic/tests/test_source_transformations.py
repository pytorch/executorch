# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from examples.models.supertonic.export import common
from examples.models.supertonic.model.config import TTSConfig
from examples.models.supertonic.model.layers import SamePad1d
from examples.models.supertonic.model.text_encoder import RelativeMultiHeadAttention
from examples.models.supertonic.model.vector_estimator import VectorEstimator
from examples.models.supertonic.model.vocoder import Vocoder
from examples.models.supertonic.source_transformations.mlx import (
    exportable_vector_estimator,
    MLXCausalPad1d,
    MLXRelativeMultiHeadAttention,
    MLXSamePad1d,
    replace_vocoder_causal_padding,
)


BOUNDS = common.ExportBounds(text_max=4, latent_max=3)
VALID_LENGTHS = ((1, 1), (4, 1), (1, 3), (2, 2), (4, 3))


def _config() -> TTSConfig:
    return TTSConfig.from_dict(
        {
            "tts_version": "test",
            "split": "test",
            "ttl": {"latent_dim": 2, "chunk_compress_factor": 2},
            "ae": {
                "sample_rate": 16000,
                "base_chunk_size": 4,
                "chunk_compress_factor": 1,
                "ldim": 2,
            },
            "dp": {"latent_dim": 2, "chunk_compress_factor": 2},
        }
    )


def _vector_model(config: TTSConfig) -> VectorEstimator:
    return (
        VectorEstimator(
            config,
            hidden_channels=8,
            time_dim=4,
            time_hidden_channels=8,
            num_main_blocks=1,
            main_convnext_dilations=(1,),
            post_time_dilations=(1,),
            post_text_dilations=(1,),
            final_dilations=(1,),
            text_channels=256,
            style_tokens=50,
            style_channels=256,
            attention_heads=1,
            style_attention_heads=2,
            max_positions=1000,
        )
        .eval()
        .half()
    )


def _vector_inputs_at(
    config: TTSConfig, text_length: int, latent_length: int
) -> tuple[torch.Tensor, ...]:
    values = common.example_inputs(config, BOUNDS)["vector_estimator"]
    inputs = (
        values[0][:, :, :latent_length].contiguous(),
        values[1][:, :, :text_length].contiguous(),
        values[2],
        values[3][:, :, :latent_length].contiguous(),
        values[4][:, :, :text_length].contiguous(),
        values[5],
        values[6],
    )
    common.validate_vector_inputs(inputs, config, BOUNDS)
    return inputs


@pytest.mark.parametrize(("text_length", "latent_length"), VALID_LENGTHS)
def test_vector_valid_domain_transform_matches_public_model(
    text_length: int, latent_length: int
) -> None:
    torch.manual_seed(0)
    config = _config()
    model = _vector_model(config)
    transformed = exportable_vector_estimator(model)
    inputs = _vector_inputs_at(config, text_length, latent_length)

    assert transformed.valid_domain_only is True
    torch.testing.assert_close(transformed(*inputs), model(*inputs))


def test_vector_transform_requires_host_validation_for_invalid_inputs() -> None:
    config = _config()
    model = _vector_model(config)
    transformed = exportable_vector_estimator(model)
    values = list(_vector_inputs_at(config, 2, 2))
    values[-1] = torch.zeros_like(values[-1])

    assert transformed.valid_domain_only is True
    with pytest.raises(ValueError, match="total_step must be finite and positive"):
        common.validate_vector_inputs(tuple(values), config, BOUNDS)


def test_vocoder_export_transform_preserves_causal_padding_semantics() -> None:
    config = _config()
    model = (
        Vocoder(
            config,
            decoder_channels=8,
            decoder_dilations=(),
            decoder_expansion=2,
            head_hidden_channels=8,
        )
        .eval()
        .half()
    )
    inputs = common.example_inputs(config, BOUNDS)["vocoder"]

    expected = model(*inputs)
    transformed = replace_vocoder_causal_padding(model)
    actual = transformed(*inputs)

    torch.testing.assert_close(actual, expected)
    assert isinstance(transformed.decoder.embed_pad, MLXCausalPad1d)
    assert isinstance(transformed.decoder.head.pad, MLXCausalPad1d)
    assert all(
        isinstance(block.pad, MLXCausalPad1d) for block in transformed.decoder.convnext
    )


@pytest.mark.parametrize("length", [1, 4])
def test_same_padding_export_transform_preserves_eager_semantics(
    length: int,
) -> None:
    inputs = torch.arange(length, dtype=torch.float32).reshape(1, 1, length)
    expected = SamePad1d(kernel_size=5, dilation=2)(inputs)
    actual = MLXSamePad1d((4, 4))(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("length", [1, 2, 5])
def test_relative_attention_export_transform_preserves_eager_semantics(
    length: int,
) -> None:
    torch.manual_seed(0)
    attention = RelativeMultiHeadAttention(
        channels=8, num_heads=2, window_size=2
    ).eval()
    inputs = torch.randn(1, 8, length)
    mask = torch.ones(1, 1, length, length)

    expected = attention(inputs, mask)
    actual = MLXRelativeMultiHeadAttention.from_attention(attention)(inputs, mask)

    torch.testing.assert_close(actual, expected)
