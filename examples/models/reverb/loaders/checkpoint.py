# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loads the released Reverb checkpoint and the Higgs audio codec."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..model.backbone import ReverbBackbone
from ..model.config import CODEBOOKS, CODES, ReverbConfig

VARIANTS = ("fp32", "fp16")


@dataclass(frozen=True)
class ReverbAssets:
    """Files of a Reverb-TTS release downloaded with `hf download`."""

    weights: Path
    config: Path
    tokenizer: Path
    voices: Path


def resolve_assets(asset_dir: str | Path, variant: str = "fp32") -> ReverbAssets:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    asset_dir = Path(asset_dir)
    folder = asset_dir / "model" / "safetensors" / variant
    assets = ReverbAssets(
        weights=folder / "model.safetensors",
        config=folder / "config.json",
        tokenizer=asset_dir / "model" / "tokenizer.json",
        voices=asset_dir / "voices",
    )
    missing = [
        path
        for path in (assets.weights, assets.config, assets.tokenizer)
        if not path.is_file()
    ]
    if missing:
        relative = ", ".join(str(path.relative_to(asset_dir)) for path in missing)
        raise FileNotFoundError(f"missing Reverb assets: {relative}")
    return assets


def load_state_dict(
    model: ReverbBackbone, weights: dict[str, torch.Tensor]
) -> ReverbBackbone:
    """Loads checkpoint tensors into `model`, centring each codebook's head rows.

    Subtracting a codebook's mean head row from all of its rows leaves its
    softmax unchanged and keeps the logits small enough for fp16.
    """
    weights = {key: value.float() for key, value in weights.items()}
    weights.pop("codebook_layer_offsets", None)
    heads = weights.pop("audio_heads.weight").view(CODEBOOKS, CODES, -1)
    heads = (heads - heads.mean(1, keepdim=True)).flatten(0, 1)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    expected_missing = {
        f"audio_heads.{name}.weight" for name in ("main", "middle", "late")
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the backbone; missing={sorted(set(missing) - expected_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.audio_heads.load_full(heads)
    return model


def load_backbone(
    asset_dir: str | Path,
    *,
    variant: str = "fp32",
    dtype: torch.dtype = torch.float32,
) -> ReverbBackbone:
    assets = resolve_assets(asset_dir, variant)
    weights = load_file(assets.weights)
    config = ReverbConfig.from_json(assets.config)
    table = weights.get("llm.embed_tokens.table.weight")
    config = replace(config, embedding_rank=None if table is None else table.shape[1])
    with torch.device("meta"):
        model = ReverbBackbone(config)
    model = model.to_empty(device="cpu")
    model.init_buffers()
    return load_state_dict(model, weights).to(dtype).eval()


def load_codec(codec_dir: str | Path):
    """The Higgs audio codec (`HiggsAudioV2TokenizerModel`), in fp32 and eval mode."""
    from transformers import HiggsAudioV2TokenizerModel

    codec = HiggsAudioV2TokenizerModel.from_pretrained(str(codec_dir)).eval()
    return codec.float()
