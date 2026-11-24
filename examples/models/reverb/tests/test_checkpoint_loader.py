# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ..loaders.checkpoint import load_backbone, resolve_assets
from ..model.config import CODEBOOKS, CODES
from .common import checkpoint_dict, random_backbone, tiny_config


def _write_release(root: Path, variant: str = "fp32") -> torch.nn.Module:
    """A fake Reverb-TTS download holding a tiny random checkpoint."""
    config = tiny_config()
    model = random_backbone(config)
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("audio_heads.")
    }
    heads = torch.cat(
        [
            model.audio_heads.main.weight,
            model.audio_heads.middle.weight,
            model.audio_heads.late.weight,
        ]
    )
    # The release stores uncentred heads and the codebook offsets.
    state["audio_heads.weight"] = (
        heads + torch.arange(CODEBOOKS).repeat_interleave(CODES)[:, None]
    )
    state["codebook_layer_offsets"] = CODES * torch.arange(CODEBOOKS)
    folder = root / "model" / "safetensors" / variant
    folder.mkdir(parents=True)
    save_file(
        {k: v.contiguous() for k, v in state.items()}, folder / "model.safetensors"
    )
    (folder / "config.json").write_text(json.dumps(checkpoint_dict(config)))
    (root / "model" / "tokenizer.json").write_text("{}")
    return model


def test_load_backbone_centres_heads_and_keeps_softmax(tmp_path: Path) -> None:
    original = _write_release(tmp_path)
    loaded = load_backbone(tmp_path)
    heads = loaded.audio_heads.main.weight.view(2, CODES, -1)
    torch.testing.assert_close(
        heads.mean(1), torch.zeros_like(heads.mean(1)), atol=1e-5, rtol=0
    )
    ids, codes = torch.arange(5), torch.zeros(CODEBOOKS, 3, dtype=torch.long)
    tokens = torch.full((CODEBOOKS, 4), 1024)
    with torch.no_grad():
        expected = original.step(tokens, original.prefill(ids, codes))
        actual = loaded.step(tokens, loaded.prefill(ids, codes))
    torch.testing.assert_close(actual[0], expected[0], atol=1e-4, rtol=1e-4)
    assert torch.equal(actual[1], expected[1])


def test_load_backbone_casts_to_the_requested_dtype(tmp_path: Path) -> None:
    _write_release(tmp_path)
    loaded = load_backbone(tmp_path, dtype=torch.float16)
    assert all(p.dtype == torch.float16 for p in loaded.parameters())
    assert loaded.inv_freq.dtype == torch.float32


def test_resolve_assets_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        resolve_assets(tmp_path)
    with pytest.raises(ValueError, match="variant"):
        resolve_assets(tmp_path, "int4")


@pytest.mark.skipif(
    not os.environ.get("REVERB_ASSET_DIR"),
    reason="set REVERB_ASSET_DIR to a Reverb-TTS download to test the release",
)
def test_release_checkpoint_loads() -> None:
    model = load_backbone(os.environ["REVERB_ASSET_DIR"])
    assert model.config.num_layers == 24
    assert model.config.embedding_rank == 256
