# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import numpy as np
import pytest

from examples.models.supertonic.loaders.voice_style_loader import load_voice_style
from examples.models.supertonic.model.config import TTSConfig


def test_tts_config_parses_published_runtime_values(tmp_path) -> None:
    config_path = tmp_path / "tts.json"
    config_path.write_text(
        json.dumps(
            {
                "tts_version": "v1.7.3",
                "split": "opensource-multilingual",
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

    config = TTSConfig.from_json(config_path)

    assert config.tts_version == "v1.7.3"
    assert config.split == "opensource-multilingual"
    assert config.ttl.latent_dim == 24
    assert config.ttl.chunk_compress_factor == 6
    assert config.ae.sample_rate == 44100
    assert config.ae.base_chunk_size == 512
    assert config.ae.chunk_compress_factor == 1
    assert config.ae.latent_dim == 24
    assert config.dp.latent_dim == 24
    assert config.dp.chunk_compress_factor == 6


def test_voice_style_loader_batches_published_dimensions(tmp_path) -> None:
    style_paths = []
    for index in range(2):
        style_path = tmp_path / f"style-{index}.json"
        style_path.write_text(
            json.dumps(
                {
                    "style_ttl": {
                        "dims": [1, 50, 256],
                        "data": [float(index)] * (50 * 256),
                    },
                    "style_dp": {
                        "dims": [1, 8, 16],
                        "data": [float(index + 2)] * (8 * 16),
                    },
                }
            ),
            encoding="utf-8",
        )
        style_paths.append(style_path)

    style = load_voice_style(style_paths)

    assert style.ttl.shape == (2, 50, 256)
    assert style.dp.shape == (2, 8, 16)
    assert style.ttl.dtype == np.float32
    assert style.dp.dtype == np.float32
    np.testing.assert_array_equal(style.ttl[:, 0, 0], [0.0, 1.0])
    np.testing.assert_array_equal(style.dp[:, 0, 0], [2.0, 3.0])


def test_voice_style_loader_rejects_unpublished_dimensions(tmp_path) -> None:
    style_path = tmp_path / "invalid-style.json"
    style_path.write_text(
        json.dumps(
            {
                "style_ttl": {
                    "dims": [1, 49, 256],
                    "data": [0.0] * (49 * 256),
                },
                "style_dp": {"dims": [1, 8, 16], "data": [0.0] * (8 * 16)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"style_ttl\.dims"):
        load_voice_style([style_path])


def test_voice_style_loader_validates_every_style_file(tmp_path) -> None:
    style_paths = []
    for name, ttl_dims in (
        ("valid", [1, 50, 256]),
        ("invalid", [1, 25, 512]),
    ):
        style_path = tmp_path / f"{name}.json"
        style_path.write_text(
            json.dumps(
                {
                    "style_ttl": {
                        "dims": ttl_dims,
                        "data": [0.0] * (50 * 256),
                    },
                    "style_dp": {
                        "dims": [1, 8, 16],
                        "data": [0.0] * (8 * 16),
                    },
                }
            ),
            encoding="utf-8",
        )
        style_paths.append(style_path)

    with pytest.raises(ValueError, match=r"style_ttl\.dims"):
        load_voice_style(style_paths)


def _write_valid_style(path: Path, value: float) -> None:
    path.write_text(
        json.dumps(
            {
                "style_ttl": {
                    "dims": [1, 50, 256],
                    "data": [value] * (50 * 256),
                },
                "style_dp": {
                    "dims": [1, 8, 16],
                    "data": [value] * (8 * 16),
                },
            }
        ),
        encoding="utf-8",
    )


def test_voice_style_loader_rejects_empty_paths() -> None:
    with pytest.raises(ValueError, match="at least one voice style path"):
        load_voice_style([])


def test_voice_style_loader_reads_each_file_once(tmp_path, monkeypatch) -> None:
    style_paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for index, style_path in enumerate(style_paths):
        _write_valid_style(style_path, float(index))

    opened_paths = []
    original_open = Path.open

    def tracking_open(path, *args, **kwargs):
        opened_paths.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    load_voice_style(style_paths)

    assert opened_paths == style_paths
