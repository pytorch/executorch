#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify cycle improvement from convert_conv_to_fc optimization (D96932767).

Compiles nn.Linear layers through the ExecuTorch ARM backend to TOSA,
then runs Vela/Regor with --verbose-performance to show per-layer cycle
counts. The optimization converts 1x1 Conv2D (produced by DecomposeLinearPass)
to FullyConnected, switching from ConvolutionMxN to VectorProduct NPU block.

Usage (must be on a Linux devserver — Regor requires Linux):

    buck run fbcode//executorch/backends/arm/test/misc:test_conv_to_fc_cycles

To compare WITHOUT the optimization (baseline), temporarily comment out
`convert_conv_to_fc` from op_rewrite_list in tosa_graph_optimiser.py and
revert the graphir_optimiser.cpp changes, then re-run.

Expected output with optimization:
    NNG Operator = FullyConnected, Op Cycles ~ 1,341, Util% (MAC) ~ 5%

Expected output WITHOUT optimization:
    NNG Operator = Conv2D,          Op Cycles ~ 9,858, Util% (MAC) ~ 0.67%
"""

import tempfile

import torch
import torch.nn as nn

from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.util._factory import create_partitioner
from executorch.exir import to_edge_transform_and_lower
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


class SimpleLinear(nn.Module):
    """Single nn.Linear layer — DecomposeLinearPass converts this to 1x1 Conv2D."""

    def __init__(self, in_features: int = 128, out_features: int = 64):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def compile_and_report(in_features: int = 128, out_features: int = 64):
    """Export, quantize, lower to TOSA, compile with Vela, print performance."""
    model = SimpleLinear(in_features, out_features).eval()
    example_input = (torch.randn(1, in_features),)

    # Export
    exported = torch.export.export(model, example_input, strict=True)

    # Quantize
    quantizer = EthosUQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(exported, quantizer)
    prepared(*example_input)
    quantized = convert_pt2e(prepared)

    # Build compile spec with verbose-performance and intermediate artifact dump
    tmpdir = tempfile.mkdtemp(prefix="conv_fc_cycles_")
    compile_spec = (
        EthosUCompileSpec(
            "ethos-u55-128",
            system_config="Ethos_U55_High_End_Embedded",
            memory_mode="Shared_Sram",
            extra_flags=["--arena-cache-size=2097152", "--verbose-performance"],
        )
        .dump_intermediate_artifacts_to(tmpdir)
    )

    partitioner = create_partitioner(compile_spec)

    print(f"Intermediate artifacts: {tmpdir}")
    print(f"Model: nn.Linear({in_features}, {out_features})")
    print()

    # Lower and compile — Vela verbose-performance output goes to stdout
    print("=" * 70)
    print("Vela --verbose-performance output:")
    print("=" * 70)
    to_edge_transform_and_lower(
        quantized,
        partitioner=[partitioner],
    )
    print("=" * 70)
    print()
    print("Look for the Conv2D/FullyConnected layer in the table above.")
    print("  WITH optimization:    NNG Operator = FullyConnected, ~1,341 cycles")
    print("  WITHOUT optimization: NNG Operator = Conv2D,         ~9,858 cycles")


def main():
    compile_and_report()


if __name__ == "__main__":
    main()
