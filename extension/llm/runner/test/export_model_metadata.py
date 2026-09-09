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
from executorch.extension.llm.export.model_metadata import model_constant_methods


class Identity(torch.nn.Module):
    """Minimal model used to serialize metadata fixtures."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return value


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
            constant_methods=model_constant_methods(
                max_context_len=4096,
                logits_to_keep=mode,
                activation_dtype=dtype,
                vocab_size=128256,
                max_seq_len=512,
            ),
        ).to_executorch()
        (output_dir / f"ModelMetadata_{mode}.pte").write_bytes(program.buffer)

    for name, invalid_field in (
        ("invalid_context", "get_max_context_len"),
        ("invalid_prefill", "get_max_seq_len"),
        ("invalid_vocab", "get_vocab_size"),
    ):
        methods = model_constant_methods(
            max_context_len=4096,
            logits_to_keep="full",
            activation_dtype="fp32",
            vocab_size=128256,
            max_seq_len=512,
        )
        methods[invalid_field] = 0
        program = to_edge(exported, constant_methods=methods).to_executorch()
        (output_dir / f"ModelMetadata_{name}.pte").write_bytes(program.buffer)

    # No constant methods: exercises rejection of missing required fields.
    missing_program = to_edge(exported).to_executorch()
    (output_dir / "ModelMetadata_missing.pte").write_bytes(missing_program.buffer)


if __name__ == "__main__":
    main()
