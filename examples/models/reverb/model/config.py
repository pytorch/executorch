# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CODEBOOKS = 8  # codec codebooks per frame, coarse (0) to fine (7)
CODES = 1025  # entries per codebook: 1024 codes and the mask
MASK = 1024  # the id of a masked slot, in every codebook
MAIN_CODEBOOKS = 2  # codebooks decided by the main passes


@dataclass(frozen=True)
class ReverbConfig:
    """The backbone's shape, read from a Reverb checkpoint's config.json."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    embedding_rank: int | None = None

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReverbConfig":
        llm = data["llm_config"]
        if data.get("num_audio_codebook", CODEBOOKS) != CODEBOOKS:
            raise ValueError(f"expected {CODEBOOKS} audio codebooks")
        if data.get("audio_vocab_size", CODES) != CODES:
            raise ValueError(f"expected {CODES} entries per codebook")
        if data.get("quantization"):
            raise ValueError(
                "quantized checkpoints are not supported; use fp32 or fp16"
            )
        rope = llm.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("only the default RoPE is supported")
        architecture = (data.get("reverb") or {}).get("architecture") or {}
        return cls(
            vocab_size=llm["vocab_size"],
            hidden_size=llm["hidden_size"],
            intermediate_size=llm["intermediate_size"],
            num_layers=llm["num_hidden_layers"],
            num_heads=llm["num_attention_heads"],
            num_kv_heads=llm["num_key_value_heads"],
            head_dim=llm["head_dim"],
            rms_norm_eps=llm["rms_norm_eps"],
            rope_theta=rope.get("rope_theta", llm.get("rope_theta", 10000.0)),
            embedding_rank=architecture.get("embedding_rank"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReverbConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
