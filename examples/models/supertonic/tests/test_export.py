# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest
import torch
from torch import nn

from examples.models.supertonic.export import common
from examples.models.supertonic.loaders import checkpoint_loader
from examples.models.supertonic.model.config import TTSConfig


def _config() -> TTSConfig:
    return TTSConfig.from_dict(
        {
            "tts_version": "test",
            "split": "test",
            "ttl": {"latent_dim": 24, "chunk_compress_factor": 6},
            "ae": {
                "sample_rate": 44100,
                "base_chunk_size": 512,
                "chunk_compress_factor": 1,
                "ldim": 24,
            },
            "dp": {"latent_dim": 24, "chunk_compress_factor": 6},
        }
    )


def test_method_set_and_public_contracts_are_exact() -> None:
    contracts = common.method_contracts(_config())

    assert set(contracts) == {
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
    }
    assert contracts["duration_predictor"] == common.MethodContract(
        ("text_ids", "style_dp", "text_mask"), "duration", ("B",), torch.float16
    )
    assert contracts["text_encoder"] == common.MethodContract(
        ("text_ids", "style_ttl", "text_mask"),
        "text_emb",
        ("B", 256, "T"),
        torch.float16,
    )
    assert contracts["vector_estimator"] == common.MethodContract(
        (
            "noisy_latent",
            "text_emb",
            "style_ttl",
            "latent_mask",
            "text_mask",
            "current_step",
            "total_step",
        ),
        "latent",
        ("B", 144, "L"),
        torch.float16,
    )
    assert contracts["vocoder"] == common.MethodContract(
        ("latent",), "waveform", ("B", "L*3072"), torch.float16
    )


def test_dynamic_shapes_share_only_binding_length_dimensions() -> None:
    shapes = common.dynamic_shapes(common.ExportBounds(31, 47))

    duration_text = shapes["duration_predictor"][0][1]
    assert duration_text is shapes["duration_predictor"][2][2]
    assert (duration_text.__name__, duration_text.min, duration_text.max) == (
        "duration_text_length",
        1,
        31,
    )

    encoder_text = shapes["text_encoder"][0][1]
    assert encoder_text is shapes["text_encoder"][2][2]
    assert (encoder_text.__name__, encoder_text.min, encoder_text.max) == (
        "encoder_text_length",
        1,
        31,
    )

    vector_latent = shapes["vector_estimator"][0][2]
    vector_text = shapes["vector_estimator"][1][2]
    assert vector_latent is shapes["vector_estimator"][3][2]
    assert vector_text is shapes["vector_estimator"][4][2]
    assert vector_latent is not vector_text
    assert (vector_latent.__name__, vector_latent.min, vector_latent.max) == (
        "vector_latent_length",
        1,
        47,
    )
    assert (vector_text.__name__, vector_text.min, vector_text.max) == (
        "vector_text_length",
        1,
        31,
    )

    vocoder_latent = shapes["vocoder"][0][2]
    assert (vocoder_latent.__name__, vocoder_latent.min, vocoder_latent.max) == (
        "vocoder_latent_length",
        1,
        47,
    )


@pytest.mark.parametrize(
    ("text_max", "latent_max", "error"),
    [
        (1, 8, "text maximum must be at least 2"),
        (8, 1, "latent maximum must be at least 2"),
        (1001, 8, "text maximum must not exceed 1000"),
        (8, 1001, "latent maximum must not exceed 1000"),
    ],
)
def test_invalid_dynamic_bounds_are_rejected_before_export(
    text_max: int, latent_max: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        common.ExportBounds(text_max, latent_max)


def test_example_inputs_are_deterministic_batch_one_fp16_with_integer_ids() -> None:
    bounds = common.ExportBounds(text_max=11, latent_max=13)

    first = common.example_inputs(_config(), bounds)
    second = common.example_inputs(_config(), bounds)

    assert set(first) == set(common.method_contracts(_config()))
    for method_name in first:
        assert len(first[method_name]) == len(second[method_name])
        for actual, repeated in zip(first[method_name], second[method_name]):
            torch.testing.assert_close(actual, repeated)
            assert actual.shape[0] == 1

    assert first["duration_predictor"][0].dtype == torch.int64
    assert first["text_encoder"][0].dtype == torch.int64
    for method_name, inputs in first.items():
        for input_name, value in zip(
            common.method_contracts(_config())[method_name].input_names, inputs
        ):
            if input_name != "text_ids":
                assert value.dtype == torch.float16

    assert first["duration_predictor"][0].shape == (1, 11)
    assert first["text_encoder"][2].shape == (1, 1, 11)
    assert first["vector_estimator"][0].shape == (1, 144, 13)
    assert first["vector_estimator"][1].shape == (1, 256, 11)
    assert first["vocoder"][0].shape == (1, 144, 13)


def test_example_inputs_cross_the_public_vector_validation_boundary(
    monkeypatch,
) -> None:
    calls = []

    def record_validation(inputs, config, bounds) -> None:
        calls.append((inputs, config, bounds))

    monkeypatch.setattr(common, "validate_vector_inputs", record_validation)
    config = _config()
    bounds = common.ExportBounds(11, 13)

    samples = common.example_inputs(config, bounds)

    assert calls == [(samples["vector_estimator"], config, bounds)]


def _valid_vector_inputs() -> tuple[torch.Tensor, ...]:
    config = _config()
    return common.example_inputs(config, common.ExportBounds(4, 3))["vector_estimator"]


def test_example_inputs_reject_flow_steps_the_native_runner_cannot_execute() -> None:
    with pytest.raises(ValueError, match="flow steps must be 5"):
        common.example_inputs(
            _config(), common.ExportBounds(4, 3), flow_steps=4
        )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda values: values[:-1], "exactly 7 tensors"),
        (
            lambda values: (values[0].repeat(2, 1, 1), *values[1:]),
            "batch size must be 1",
        ),
        (
            lambda values: (
                values[0],
                values[1].float(),
                *values[2:],
            ),
            "text_emb must have dtype torch.float16",
        ),
        (
            lambda values: (
                values[0][:, :, :0],
                values[1],
                values[2],
                values[3][:, :, :0],
                *values[4:],
            ),
            "latent length must be in",
        ),
        (
            lambda values: (
                values[0],
                values[1][:, :, :0],
                values[2],
                values[3],
                values[4][:, :, :0],
                *values[5:],
            ),
            "text length must be in",
        ),
        (
            lambda values: (
                torch.cat((values[0], values[0][:, :, :1]), dim=2),
                values[1],
                values[2],
                torch.cat((values[3], values[3][:, :, :1]), dim=2),
                *values[4:],
            ),
            "latent length must be in",
        ),
        (
            lambda values: (
                values[0],
                values[1],
                values[2],
                torch.zeros_like(values[3]),
                *values[4:],
            ),
            "latent_mask must contain a valid position",
        ),
        (
            lambda values: (
                *values[:6],
                torch.zeros_like(values[6]),
            ),
            "total_step must be finite and positive",
        ),
        (
            lambda values: (
                values[0][:, :, ::2],
                *values[1:],
            ),
            "noisy_latent must be contiguous",
        ),
    ],
)
def test_public_vector_validator_rejects_invalid_pte_domain_inputs(
    mutate, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        common.validate_vector_inputs(
            mutate(_valid_vector_inputs()),
            _config(),
            common.ExportBounds(4, 3),
        )


def test_text_vocabulary_size_requires_matching_model_embeddings() -> None:
    class ModelPair(nn.Module):
        def __init__(self, vocabulary_size: int) -> None:
            super().__init__()
            self.text_embedder = nn.Module()
            self.text_embedder.char_embedder = nn.Embedding(vocabulary_size, 2)

    duration = nn.Module()
    duration.sentence_encoder = ModelPair(17)
    encoder = nn.Module()
    encoder.text_encoder = ModelPair(17)
    models = {"duration_predictor": duration, "text_encoder": encoder}

    assert common.text_vocabulary_size(models) == 17
    encoder.text_encoder = ModelPair(18)
    with pytest.raises(ValueError, match="must use the same vocabulary"):
        common.text_vocabulary_size(models)


def test_runtime_metadata_contains_host_pipeline_constants() -> None:
    metadata = common.runtime_metadata(
        _config(),
        common.ExportBounds(31, 47),
        text_vocabulary_size=8322,
        flow_steps=5,
    )

    assert metadata == {
        "get_sample_rate": 44100,
        "get_base_chunk_size": 512,
        "get_chunk_compress_factor": 6,
        "get_flow_steps": 5,
        "get_text_vocabulary_size": 8322,
        "get_latent_dim": 24,
        "get_latent_channels": 144,
        "get_max_text_length": 31,
        "get_max_latent_length": 47,
        "get_batch_size": 1,
        "get_activation_dtype": "float16",
        "enable_dynamic_shape": True,
    }

    with pytest.raises(ValueError, match="flow steps must be 5"):
        common.runtime_metadata(
            _config(),
            common.ExportBounds(),
            text_vocabulary_size=8322,
            flow_steps=4,
        )
    with pytest.raises(ValueError, match="text vocabulary size must be positive"):
        common.runtime_metadata(
            _config(), common.ExportBounds(), text_vocabulary_size=0
        )


def test_asset_paths_follow_the_published_layout_without_downloads(tmp_path) -> None:
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "tts.json").write_text(
        json.dumps(
            {
                "tts_version": "test",
                "split": "test",
                "ttl": {"latent_dim": 24, "chunk_compress_factor": 6},
                "ae": {
                    "sample_rate": 44100,
                    "base_chunk_size": 512,
                    "chunk_compress_factor": 1,
                    "ldim": 24,
                },
                "dp": {"latent_dim": 24, "chunk_compress_factor": 6},
            }
        ),
        encoding="utf-8",
    )
    for method_name in common.METHOD_NAMES:
        (onnx_dir / f"{method_name}.onnx").touch()

    assets = common.resolve_assets(tmp_path)

    assert assets.config == onnx_dir / "tts.json"
    assert assets.models == {
        name: onnx_dir / f"{name}.onnx" for name in common.METHOD_NAMES
    }


def test_asset_paths_report_every_missing_required_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        common.resolve_assets(tmp_path)

    message = str(error.value)
    assert "onnx/tts.json" in message
    for method_name in common.METHOD_NAMES:
        assert f"onnx/{method_name}.onnx" in message


def test_load_models_constructs_every_stage_from_the_same_config(
    tmp_path, monkeypatch
) -> None:
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "tts.json").write_text(
        json.dumps(
            {
                "tts_version": "test",
                "split": "test",
                "ttl": {"latent_dim": 24, "chunk_compress_factor": 6},
                "ae": {
                    "sample_rate": 44100,
                    "base_chunk_size": 512,
                    "chunk_compress_factor": 1,
                    "ldim": 24,
                },
                "dp": {"latent_dim": 24, "chunk_compress_factor": 6},
            }
        ),
        encoding="utf-8",
    )
    calls = []
    expected_models = {}
    for method_name in common.METHOD_NAMES:
        model_path = onnx_dir / f"{method_name}.onnx"
        model_path.touch()
        model = nn.Linear(1, 1)
        model.train()
        expected_models[method_name] = model

        def fake_loader(path, config, *, _name=method_name, _model=model):
            calls.append((_name, path, config))
            return _model

        monkeypatch.setattr(checkpoint_loader, f"load_{method_name}", fake_loader)

    config, models = common.load_models(tmp_path)

    assert config == _config()
    assert models == expected_models
    assert [name for name, _, _ in calls] == list(common.METHOD_NAMES)
    assert all(path == onnx_dir / f"{name}.onnx" for name, path, _ in calls)
    assert all(loaded_config is config for _, _, loaded_config in calls)
    assert all(not model.training for model in models.values())


def test_fp16_conversion_preserves_integer_tensors() -> None:
    model = nn.Linear(2, 3)
    model.register_buffer("float_buffer", torch.ones(2, dtype=torch.float32))
    model.register_buffer("integer_buffer", torch.ones(2, dtype=torch.int64))

    converted = common.convert_models_to_fp16({"stage": model})

    assert converted["stage"] is model
    assert model.weight.dtype == torch.float16
    assert model.bias.dtype == torch.float16
    assert model.float_buffer.dtype == torch.float16
    assert model.integer_buffer.dtype == torch.int64
