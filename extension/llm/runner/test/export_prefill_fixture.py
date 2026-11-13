#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass


class PrefillFixture(torch.nn.Module):
    """Logits expose the call count, token sum, and supplied start position."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("calls", torch.zeros(1))
        self.register_buffer("vocab", torch.arange(32, dtype=torch.float32).view(1, 32))

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        self.calls.add_(1)
        peak = 2 * tokens.to(torch.float32).sum() + input_pos + self.calls
        distance = self.vocab - peak
        return -(distance * distance)


def main() -> None:
    """Export bounded-dynamic and single-token static runner fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dynamic in (("dynamic", True), ("static", False)):
        exported = torch.export.export(
            PrefillFixture(),
            (
                torch.ones(1, 3 if dynamic else 1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            ),
            dynamic_shapes=(
                ({1: torch.export.Dim("tokens", min=1, max=7)}, {}) if dynamic else None
            ),
            strict=True,
        )
        program = to_edge(
            exported,
            constant_methods={
                "get_max_seq_len": 8,
                "get_max_context_len": 16,
                "enable_dynamic_shape": dynamic,
                "use_kv_cache": True,
            },
        ).to_executorch(
            ExecutorchBackendConfig(passes=[InitializedMutableBufferPass(["calls"])])
        )
        (output_dir / f"PrefillFixture_{name}.pte").write_bytes(program.buffer)


if __name__ == "__main__":
    main()
