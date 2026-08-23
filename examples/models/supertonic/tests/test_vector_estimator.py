# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch

from examples.models.supertonic.model.config import TTSConfig


def _vector_estimator():
    return importlib.import_module("examples.models.supertonic.model.vector_estimator")


def _config(
    *,
    latent_dim: int = 2,
    compress_factor: int = 2,
    base_chunk_size: int = 4,
) -> TTSConfig:
    return TTSConfig.from_dict(
        {
            "tts_version": "test",
            "split": "test",
            "ttl": {
                "latent_dim": latent_dim,
                "chunk_compress_factor": compress_factor,
            },
            "ae": {
                "sample_rate": 16000,
                "base_chunk_size": base_chunk_size,
                "chunk_compress_factor": 1,
                "ldim": latent_dim,
            },
            "dp": {
                "latent_dim": latent_dim,
                "chunk_compress_factor": compress_factor,
            },
        }
    )


def _small_model():
    return (
        _vector_estimator()
        .VectorEstimator(
            _config(),
            hidden_channels=8,
            time_dim=4,
            time_hidden_channels=8,
            num_main_blocks=1,
            main_convnext_dilations=(1,),
            post_time_dilations=(1,),
            post_text_dilations=(1,),
            final_dilations=(1,),
            text_channels=6,
            style_tokens=3,
            style_channels=6,
            attention_heads=2,
            style_attention_heads=2,
            max_positions=16,
        )
        .eval()
    )


def _inputs():
    return (
        torch.randn(2, 4, 5),
        torch.randn(2, 6, 4),
        torch.randn(2, 3, 6),
        torch.tensor(
            [
                [[1.0, 1.0, 1.0, 1.0, 0.0]],
                [[1.0, 1.0, 1.0, 0.0, 0.0]],
            ]
        ),
        torch.tensor(
            [
                [[1.0, 1.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0, 0.0]],
            ]
        ),
        torch.tensor([1.0, 2.0]),
        torch.tensor([4.0, 4.0]),
    )


def test_vector_estimator_returns_finite_masked_latent_deterministically() -> None:
    torch.manual_seed(0)
    model = _small_model()
    inputs = _inputs()

    first = model(*inputs)
    repeated = model(*inputs)

    assert first.shape == (2, 4, 5)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(
        first * (1.0 - inputs[3]),
        torch.zeros_like(first),
    )


def test_vector_estimator_ignores_masked_latent_and_text_values() -> None:
    torch.manual_seed(1)
    model = _small_model()
    inputs = list(_inputs())
    changed = [value.clone() for value in inputs]
    changed[0] = torch.where(
        inputs[3] == 0,
        torch.full_like(inputs[0], 1000.0),
        inputs[0],
    )
    changed[1] = torch.where(
        inputs[4] == 0,
        torch.full_like(inputs[1], -1000.0),
        inputs[1],
    )

    torch.testing.assert_close(model(*inputs), model(*changed))


def test_vector_estimator_uses_current_and_total_steps() -> None:
    torch.manual_seed(2)
    model = _small_model()
    inputs = list(_inputs())
    changed_current = [value.clone() for value in inputs]
    changed_current[5] = inputs[5] + 1.0
    changed_total = [value.clone() for value in inputs]
    changed_total[6] = inputs[6] * 2.0

    baseline = model(*inputs)

    assert not torch.allclose(baseline, model(*changed_current))
    assert not torch.allclose(baseline, model(*changed_total))


@pytest.mark.parametrize(
    ("transform", "error"),
    [
        (
            lambda values: [value[:0] for value in values],
            "batch size must be positive",
        ),
        (
            lambda values: [
                values[0][:, :, :0],
                values[1],
                values[2],
                values[3][:, :, :0],
                values[4],
                values[5],
                values[6],
            ],
            "latent length must be positive",
        ),
        (
            lambda values: [
                values[0],
                values[1][:, :, :0],
                values[2],
                values[3],
                values[4][:, :, :0],
                values[5],
                values[6],
            ],
            "text length must be positive",
        ),
        (
            lambda values: [
                values[0],
                values[1],
                values[2],
                torch.cat(
                    (torch.zeros_like(values[3][:1]), values[3][1:]),
                    dim=0,
                ),
                values[4],
                values[5],
                values[6],
            ],
            "latent_mask must contain a valid position per sample",
        ),
        (
            lambda values: [
                values[0],
                values[1],
                values[2],
                values[3],
                torch.cat(
                    (torch.zeros_like(values[4][:1]), values[4][1:]),
                    dim=0,
                ),
                values[5],
                values[6],
            ],
            "text_mask must contain a valid position per sample",
        ),
        (
            lambda values: [
                values[0],
                values[1],
                values[2],
                torch.full_like(values[3], float("nan")),
                values[4],
                values[5],
                values[6],
            ],
            "latent_mask must contain a valid position per sample",
        ),
        (
            lambda values: [
                values[0],
                values[1],
                values[2],
                values[3],
                torch.full_like(values[4], float("inf")),
                values[5],
                values[6],
            ],
            "text_mask must contain a valid position per sample",
        ),
        (
            lambda values: values[:6] + [torch.tensor([0.0, 4.0])],
            "total_step must be finite and positive",
        ),
        (
            lambda values: values[:6] + [torch.tensor([-1.0, 4.0])],
            "total_step must be finite and positive",
        ),
        (
            lambda values: values[:6] + [torch.tensor([float("inf"), 4.0])],
            "total_step must be finite and positive",
        ),
        (
            lambda values: values[:6] + [torch.tensor([float("nan"), 4.0])],
            "total_step must be finite and positive",
        ),
        (
            lambda values: values[:5] + [torch.tensor([float("inf"), 2.0]), values[6]],
            "current_step must be finite",
        ),
        (
            lambda values: values[:5] + [torch.tensor([float("nan"), 2.0]), values[6]],
            "current_step must be finite",
        ),
    ],
)
def test_vector_estimator_rejects_degenerate_public_inputs_before_operators(
    transform, error: str
) -> None:
    model = _small_model()
    inputs = transform(list(_inputs()))

    with pytest.raises(ValueError, match=error):
        model(*inputs)


@pytest.mark.parametrize(
    ("index", "replacement", "error"),
    [
        (0, torch.zeros(144, 3), r"noisy_latent.*\[B, 144, L\]"),
        (0, torch.zeros(1, 143, 3), r"noisy_latent.*\[B, 144, L\]"),
        (1, torch.zeros(1, 255, 4), r"text_emb.*\[B, 256, T\]"),
        (2, torch.zeros(1, 49, 256), r"style_ttl.*\[B, 50, 256\]"),
        (3, torch.ones(1, 2, 3), r"latent_mask.*\[B, 1, L\]"),
        (4, torch.ones(1, 1, 3), "text lengths must match"),
        (5, torch.zeros(1, 1), r"current_step.*\[B\]"),
        (6, torch.zeros(2), "batch sizes must match"),
    ],
)
def test_vector_estimator_validates_public_contract_before_operators(
    index: int, replacement: torch.Tensor, error: str
) -> None:
    model = _vector_estimator().VectorEstimator(
        _config(latent_dim=24, compress_factor=6),
        hidden_channels=4,
        time_dim=4,
        time_hidden_channels=4,
        num_main_blocks=0,
        main_convnext_dilations=(),
        post_time_dilations=(),
        post_text_dilations=(),
        final_dilations=(),
        text_channels=256,
        style_tokens=50,
        style_channels=256,
        attention_heads=1,
        style_attention_heads=1,
        max_positions=8,
    )
    inputs = [
        torch.zeros(1, 144, 3),
        torch.zeros(1, 256, 4),
        torch.zeros(1, 50, 256),
        torch.ones(1, 1, 3),
        torch.ones(1, 1, 4),
        torch.zeros(1),
        torch.ones(1),
    ]
    inputs[index] = replacement

    with pytest.raises(ValueError, match=error):
        model(*inputs)


def test_time_encoder_matches_exported_sinusoidal_and_mish_semantics() -> None:
    module = _vector_estimator()
    encoder = module.TimeEncoder(time_dim=4, hidden_channels=3)
    with torch.no_grad():
        encoder.mlp[0].linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
        encoder.mlp[0].linear.bias.zero_()
        encoder.mlp[2].linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            )
        )
        encoder.mlp[2].linear.bias.zero_()

    time = torch.tensor([0.25])
    frequencies = 10000.0 ** (-torch.arange(2, dtype=torch.float32))
    sinusoidal = torch.cat(
        (
            torch.sin(time[:, None] * 1000.0 * frequencies),
            torch.cos(time[:, None] * 1000.0 * frequencies),
        ),
        dim=-1,
    )
    hidden = sinusoidal[:, :3]
    mish = hidden * torch.tanh(torch.nn.functional.softplus(hidden))
    expected = torch.stack(
        (mish[:, 0], mish[:, 1], mish[:, 2], mish.sum(dim=-1)),
        dim=-1,
    ).unsqueeze(-1)

    torch.testing.assert_close(encoder(time), expected)


def test_rotary_embedding_rotates_exported_feature_halves() -> None:
    attention = _vector_estimator().RotaryCrossAttention(
        channels=4,
        context_channels=4,
        num_heads=1,
        max_positions=4,
        rotary_base=10000.0,
        rotary_scale=10.0,
    )
    values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    sine = torch.tensor([[[0.5, -0.25]]])
    cosine = torch.tensor([[[0.75, 0.125]]])

    rotated = attention._apply_rotary(values, sine, cosine)

    torch.testing.assert_close(
        rotated,
        torch.tensor([[[[-0.75, 1.25, 2.75, 0.0]]]]),
    )


def test_cfg_uses_conditional_then_unconditional_batches_and_4_minus_3() -> None:
    conditional = torch.tensor([[[1.0]], [[2.0]]])
    unconditional = torch.tensor([[[10.0]], [[20.0]]])
    vector = torch.cat((conditional, unconditional), dim=0)

    guided = _vector_estimator().VectorEstimator._apply_guidance(vector)

    torch.testing.assert_close(
        guided,
        4.0 * conditional - 3.0 * unconditional,
    )


def test_rotary_attention_uses_published_divide_by_16_score_scaling() -> None:
    attention = _vector_estimator().RotaryCrossAttention(
        channels=8,
        context_channels=256,
        num_heads=2,
        max_positions=4,
    )
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = torch.tensor([[[[4.0, 3.0, 2.0, 1.0]]]])

    scores = attention._scaled_scores(query, key)

    torch.testing.assert_close(
        scores,
        torch.matmul(query, key.transpose(-2, -1)) / 16.0,
    )


def test_rotary_positions_normalize_per_sample_and_heads_lead_batch() -> None:
    attention = _vector_estimator().RotaryCrossAttention(
        channels=8,
        context_channels=8,
        num_heads=2,
        max_positions=4,
    )
    values = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    mask = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 1.0]],
        ]
    )

    heads = attention._split_heads(values)
    sine, cosine = attention._angles(mask)
    positions = torch.tensor(
        [
            [[0.0], [0.5], [1.0], [1.5]],
            [[0.0], [0.25], [0.5], [0.75]],
        ]
    )
    expected_angles = positions * attention.theta

    assert heads.shape == (2, 2, 1, 4)
    torch.testing.assert_close(heads[0], values[:, :, :4])
    torch.testing.assert_close(heads[1], values[:, :, 4:])
    torch.testing.assert_close(sine, torch.sin(expected_angles))
    torch.testing.assert_close(cosine, torch.cos(expected_angles))


def test_rotary_attention_masks_keys_before_and_queries_after_softmax() -> None:
    attention = _vector_estimator().RotaryCrossAttention(
        channels=4,
        context_channels=4,
        num_heads=1,
        max_positions=2,
    )
    with torch.no_grad():
        attention.theta.zero_()
        for projection in (
            attention.W_query,
            attention.W_key,
            attention.W_value,
            attention.out_fc,
        ):
            projection.linear.weight.copy_(torch.eye(4))
            projection.linear.bias.zero_()
    inputs = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    context = torch.tensor([[[2.0, 3.0, 4.0, 5.0], [1000.0, 1000.0, 1000.0, 1000.0]]])
    query_mask = torch.tensor([[[1.0, 0.0]]])
    key_mask = torch.tensor([[[1.0, 0.0]]])

    output = attention(inputs, context, query_mask, key_mask)

    torch.testing.assert_close(output[:, :1], context[:, :1])
    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))
