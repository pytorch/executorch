#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch

from executorch.exir import to_edge
from executorch.extension.llm.export.model_metadata import (
    write_activation_dtype,
    write_cache_geometry,
    write_logits_to_keep_mode,
    write_max_context_len,
    write_max_seq_len,
    write_vocab_size,
)


class Identity(torch.nn.Module):
    """Minimal model used to serialize metadata fixtures."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return value


def all_methods(logits_to_keep: str, activation_dtype: str) -> dict[str, object]:
    """Compose the full metadata set from the individual per-constant writers."""
    return {
        **write_max_context_len(4096),
        **write_vocab_size(128256),
        **write_activation_dtype(activation_dtype),
        **write_logits_to_keep_mode(logits_to_keep),
        **write_max_seq_len(512),
        **write_cache_geometry(
            kv_heads=[8, 4, 2],
            head_dims=[64, 80, 96],
            windows=[0, 512, 128],
        ),
    }


def main() -> None:
    """Generate model metadata PTE fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = torch.export.export(Identity(), (torch.ones(1),), strict=True)
    for mode, dtype in (
        ("full", "fp32"),
        ("last", "fp16"),
        ("selected", "bf16"),
    ):
        program = to_edge(
            exported,
            constant_methods=all_methods(mode, dtype),
        ).to_executorch()
        (output_dir / f"ModelMetadata_{mode}.pte").write_bytes(program.buffer)

    for name, invalid_field in (
        ("invalid_context", "get_max_context_len"),
        ("invalid_prefill", "get_max_seq_len"),
        ("invalid_vocab", "get_vocab_size"),
    ):
        methods = all_methods("full", "fp32")
        methods[invalid_field] = 0
        program = to_edge(exported, constant_methods=methods).to_executorch()
        (output_dir / f"ModelMetadata_{name}.pte").write_bytes(program.buffer)

    malformed_geometry = {
        "wrong_type": ("get_kv_heads", torch.tensor([8, 4, 2], dtype=torch.float32)),
        "mismatched": ("get_head_dims", torch.tensor([64, 80], dtype=torch.int32)),
        "invalid_heads": ("get_kv_heads", torch.tensor([8, 0, 2], dtype=torch.int32)),
        "invalid_dims": (
            "get_head_dims",
            torch.tensor([64, -1, 96], dtype=torch.int32),
        ),
        "invalid_windows": (
            "get_windows",
            torch.tensor([0, -1, 128], dtype=torch.int32),
        ),
        "empty": ("get_n_caches", 0),
    }
    for name, (field, value) in malformed_geometry.items():
        methods = all_methods("full", "fp32")
        methods[field] = value
        program = to_edge(exported, constant_methods=methods).to_executorch()
        (output_dir / f"ModelMetadata_geometry_{name}.pte").write_bytes(program.buffer)

    # No constant methods: exercises rejection of missing required fields.
    missing_program = to_edge(exported).to_executorch()
    (output_dir / "ModelMetadata_missing.pte").write_bytes(missing_program.buffer)


if __name__ == "__main__":
    main()
