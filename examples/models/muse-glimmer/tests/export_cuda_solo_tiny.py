# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exports the tiny Solo artifact that solo_pipelined_decode_test loads.

    python examples/models/muse-glimmer/tests/export_cuda_solo_tiny.py --output-dir DIR
    ET_MUSE_SOLO_TINY_DIR=DIR ctest -R muse_glimmer_solo_pipelined_decode

The artifact samples on the device, so its decode method returns an int64
token, which the pipelined decode needs. tokenizer.json is a byte-level BPE
vocabulary over the model's 256 ids.
"""

import argparse
import json
from pathlib import Path

import torch
from executorch.examples.models.muse_glimmer.export.export_solo import export_and_lower
from executorch.examples.models.muse_glimmer.model.model import (
    MuseGlimmerConfig,
    MuseGlimmerModel,
)

CONFIG = MuseGlimmerConfig(
    dim=256,
    n_layers=8,
    n_heads=4,
    n_kv_heads=2,
    head_dim=64,
    ffn_dim_multiplier=2.0,
    vocab_size=256,
    global_attn_cfg="[16,16,16,0]",
    max_seq_len=64,
)

_BYTE_LEVEL = {
    "type": "ByteLevel",
    "add_prefix_space": False,
    "trim_offsets": False,
    "use_regex": False,
}


def _byte_level_vocab() -> dict[str, int]:
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    remapped = [byte for byte in range(256) if byte not in printable]
    chars = {byte: chr(byte) for byte in printable}
    chars.update({byte: chr(256 + i) for i, byte in enumerate(remapped)})
    return {chars[byte]: byte for byte in range(256)}


def write_tokenizer(path: Path) -> None:
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": _BYTE_LEVEL,
        "post_processor": _BYTE_LEVEL,
        "decoder": _BYTE_LEVEL,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": _byte_level_vocab(),
            "merges": [],
        },
    }
    path.write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")


def export(output_dir: Path) -> None:
    torch.manual_seed(0)
    model = MuseGlimmerModel(CONFIG).to(torch.bfloat16).eval()
    model.activation_dtype = torch.bfloat16
    output_dir.mkdir(parents=True, exist_ok=True)
    export_and_lower(model, CONFIG, str(output_dir))
    write_tokenizer(output_dir / "tokenizer.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    export(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
