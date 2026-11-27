# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.models.supertonic.loaders.checkpoint_loader import (
    load_duration_predictor,
    load_text_encoder,
    load_vector_estimator,
    load_vocoder,
)
from examples.models.supertonic.model.config import TTSConfig

_REAL_MODEL_DIR = os.environ.get("SUPERTONIC_MODEL_DIR")


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    error = actual.astype(np.float64) - expected.astype(np.float64)
    signal_power = np.sum(expected.astype(np.float64) ** 2)
    noise_power = np.sum(error**2)
    cosine = np.dot(actual.reshape(-1), expected.reshape(-1)) / (
        np.linalg.norm(actual) * np.linalg.norm(expected)
    )
    return {
        "max_error": float(np.max(np.abs(error))),
        "mean_error": float(np.mean(np.abs(error))),
        "cosine": float(cosine),
        "sqnr_db": float(10.0 * np.log10(signal_power / noise_power)),
    }


@pytest.mark.skipif(
    _REAL_MODEL_DIR is None,
    reason="set SUPERTONIC_MODEL_DIR for eager PyTorch/ONNX Runtime parity",
)
def test_duration_predictor_matches_published_onnx() -> None:
    ort = pytest.importorskip("onnxruntime")
    model_dir = Path(_REAL_MODEL_DIR) / "onnx"
    config = TTSConfig.from_json(model_dir / "tts.json")
    model_path = model_dir / "duration_predictor.onnx"
    model = load_duration_predictor(model_path, config).eval()
    rng = np.random.default_rng(1234)
    inputs = {
        "text_ids": rng.integers(0, 256, size=(2, 11), dtype=np.int64),
        "style_dp": rng.standard_normal((2, 8, 16), dtype=np.float32),
        "text_mask": np.asarray(
            [
                [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
            ],
            dtype=np.float32,
        ),
    }

    expected = ort.InferenceSession(str(model_path)).run(None, inputs)[0]
    with torch.no_grad():
        actual = model(
            torch.from_numpy(inputs["text_ids"]),
            torch.from_numpy(inputs["style_dp"]),
            torch.from_numpy(inputs["text_mask"]),
        ).numpy()
    metrics = _metrics(actual, expected)
    print(f"duration predictor parity: {metrics}")

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert metrics["max_error"] < 1e-6
    assert metrics["mean_error"] < 2e-7
    assert metrics["cosine"] > 0.9999999
    assert metrics["sqnr_db"] > 120.0


@pytest.mark.skipif(
    _REAL_MODEL_DIR is None,
    reason="set SUPERTONIC_MODEL_DIR for eager PyTorch/ONNX Runtime parity",
)
def test_text_encoder_matches_published_onnx() -> None:
    ort = pytest.importorskip("onnxruntime")
    model_dir = Path(_REAL_MODEL_DIR) / "onnx"
    config = TTSConfig.from_json(model_dir / "tts.json")
    model_path = model_dir / "text_encoder.onnx"
    model = load_text_encoder(model_path, config).eval()
    rng = np.random.default_rng(5678)
    inputs = {
        "text_ids": rng.integers(0, 256, size=(2, 11), dtype=np.int64),
        "style_ttl": rng.standard_normal((2, 50, 256), dtype=np.float32),
        "text_mask": np.asarray(
            [
                [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
            ],
            dtype=np.float32,
        ),
    }

    expected = ort.InferenceSession(str(model_path)).run(None, inputs)[0]
    with torch.no_grad():
        actual = model(
            torch.from_numpy(inputs["text_ids"]),
            torch.from_numpy(inputs["style_ttl"]),
            torch.from_numpy(inputs["text_mask"]),
        ).numpy()
    metrics = _metrics(actual, expected)
    print(f"text encoder parity: {metrics}")

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert metrics["max_error"] < 1e-4
    assert metrics["mean_error"] < 5e-6
    assert metrics["cosine"] > 0.999999
    assert metrics["sqnr_db"] > 90.0


@pytest.mark.skipif(
    _REAL_MODEL_DIR is None,
    reason="set SUPERTONIC_MODEL_DIR for eager PyTorch/ONNX Runtime parity",
)
def test_vector_estimator_matches_published_onnx() -> None:
    ort = pytest.importorskip("onnxruntime")
    model_dir = Path(_REAL_MODEL_DIR) / "onnx"
    config = TTSConfig.from_json(model_dir / "tts.json")
    model_path = model_dir / "vector_estimator.onnx"
    model = load_vector_estimator(model_path, config).eval()
    rng = np.random.default_rng(9012)
    inputs = {
        "noisy_latent": rng.standard_normal((1, 144, 8), dtype=np.float32),
        "text_emb": rng.standard_normal((1, 256, 7), dtype=np.float32),
        "style_ttl": rng.standard_normal((1, 50, 256), dtype=np.float32),
        "latent_mask": np.asarray(
            [[[1, 1, 1, 1, 1, 1, 0, 0]]],
            dtype=np.float32,
        ),
        "text_mask": np.asarray(
            [[[1, 1, 1, 1, 1, 0, 0]]],
            dtype=np.float32,
        ),
        "current_step": np.asarray([2.0], dtype=np.float32),
        "total_step": np.asarray([5.0], dtype=np.float32),
    }

    expected = ort.InferenceSession(str(model_path)).run(None, inputs)[0]
    with torch.no_grad():
        actual = model(*(torch.from_numpy(value) for value in inputs.values())).numpy()
    metrics = _metrics(actual, expected)
    print(f"vector estimator parity: {metrics}")

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert metrics["max_error"] < 1e-5
    assert metrics["mean_error"] < 1e-6
    assert metrics["cosine"] > 0.999999
    assert metrics["sqnr_db"] > 110.0


@pytest.mark.skipif(
    _REAL_MODEL_DIR is None,
    reason="set SUPERTONIC_MODEL_DIR for eager PyTorch/ONNX Runtime parity",
)
def test_vocoder_matches_published_onnx() -> None:
    ort = pytest.importorskip("onnxruntime")
    model_dir = Path(_REAL_MODEL_DIR) / "onnx"
    config = TTSConfig.from_json(model_dir / "tts.json")
    model_path = model_dir / "vocoder.onnx"
    model = load_vocoder(model_path, config).eval()
    rng = np.random.default_rng(3456)
    inputs = {
        "latent": rng.standard_normal((1, 144, 3), dtype=np.float32),
    }

    expected = ort.InferenceSession(str(model_path)).run(None, inputs)[0]
    with torch.no_grad():
        actual = model(torch.from_numpy(inputs["latent"])).numpy()
    metrics = _metrics(actual, expected)
    waveform_correlation = float(
        np.corrcoef(actual.reshape(-1), expected.reshape(-1))[0, 1]
    )
    print(
        "vocoder parity: " f"{metrics | {'waveform_correlation': waveform_correlation}}"
    )

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert metrics["max_error"] < 2e-6
    assert metrics["mean_error"] < 2e-7
    assert metrics["cosine"] > 0.9999999
    assert metrics["sqnr_db"] > 105.0
    assert waveform_correlation > 0.9999999
