# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LatentConfig:
    latent_dim: int
    chunk_compress_factor: int


@dataclass(frozen=True)
class AutoencoderConfig:
    sample_rate: int
    base_chunk_size: int
    chunk_compress_factor: int
    latent_dim: int


@dataclass(frozen=True)
class TTSConfig:
    tts_version: str
    split: str
    ttl: LatentConfig
    ae: AutoencoderConfig
    dp: LatentConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TTSConfig":
        ttl = data["ttl"]
        ae = data["ae"]
        dp = data["dp"]
        return cls(
            tts_version=data["tts_version"],
            split=data["split"],
            ttl=LatentConfig(
                latent_dim=ttl["latent_dim"],
                chunk_compress_factor=ttl["chunk_compress_factor"],
            ),
            ae=AutoencoderConfig(
                sample_rate=ae["sample_rate"],
                base_chunk_size=ae["base_chunk_size"],
                chunk_compress_factor=ae["chunk_compress_factor"],
                latent_dim=ae["ldim"],
            ),
            dp=LatentConfig(
                latent_dim=dp["latent_dim"],
                chunk_compress_factor=dp["chunk_compress_factor"],
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TTSConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            return cls.from_dict(json.load(config_file))
