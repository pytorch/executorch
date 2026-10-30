# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exports the tiny packed target + DFlash draft that cuda_dflash_native_test loads.

    python examples/models/muse-glimmer/tests/export_cuda_dflash_tiny.py --output-dir DIR
    ET_MUSE_DFLASH_TINY_DIR=DIR ctest -R muse_glimmer_cuda_dflash

The geometry is the native test's contract: vocabulary 256, two tapped target
layers of width 64 (128 hidden columns), DFlash block 16, 32-token steps, and
four draft owners.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from executorch.examples.models.muse_glimmer.export.export_solo import (
    export_cuda_batching,
)
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    DFlashConfig,
    DFlashDraftModel,
)
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
from executorch.examples.models.muse_glimmer.tests.test_pipeline import TINY_CONFIG


def export(output_dir: Path) -> None:
    torch.manual_seed(123)
    config = replace(TINY_CONFIG, dim=64, n_layers=2, head_dim=16, max_seq_len=128)
    target = MuseGlimmerModel(config).to(torch.bfloat16).eval()
    target.activation_dtype = torch.bfloat16
    draft_config = DFlashConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        vocab_size=config.vocab_size,
        target_layers=[0, 1],
        block_size=16,
        mask_token_id=255,
        max_seq_len=128,
    )
    draft = DFlashDraftModel(draft_config, max_context_length=32)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_cuda_batching(
        target,
        target.config,
        str(output_dir),
        max_step_tokens=32,
        max_cells=512,
        forward_sizes=(1, 2, 4, 8, 16),
        dflash_draft=draft.to(torch.bfloat16).eval(),
        max_draft_owners=4,
    )
    print(f"Wrote {output_dir}/model.pte and aoti_cuda_blob.ptd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    export(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
