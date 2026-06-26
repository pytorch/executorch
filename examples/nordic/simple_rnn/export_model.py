#!/usr/bin/env python3
# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Export a simple RNN for the AXON NPU — multi-subgraph delegation.

This model demonstrates why real models produce multiple AXON subgraphs.
A simple RNN has Linear layers (AXON-delegatable) separated by a
recurrent hidden state update (tanh — runs on CPU). The partitioner
cannot group the FC layers into one subgraph because the recurrent
loop between them is not TOSA-compatible.

Model (single-step RNN):
    input (4-dim) + hidden (8-dim)
        → fc_ih: Linear(4 → 8)     ← AXON subgraph A
        → fc_hh: Linear(8 → 8)     ← AXON subgraph B
        → add + tanh                ← CPU (recurrent state update)
        → fc_out: Linear(8 → 2)    ← AXON subgraph C
    output (2-dim) + new_hidden (8-dim)

The tanh activation on the hidden state is not TOSA INT-delegatable
(it requires the TABLE op which breaks the subgraph boundary), so
fc_ih, fc_hh, and fc_out become separate AXON subgraphs.

Usage:
    SDK_EDGE_AI_PATH=~/sdk-edge-ai ./run_export.sh
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


class SimpleRNNStep(nn.Module):
    """Single-step RNN cell with separate input and hidden projections.

    Unrolled for export: takes input + hidden, returns output + new_hidden.
    The tanh between the Linear layers forces multi-subgraph delegation.
    """

    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        super().__init__()
        self.fc_ih = nn.Linear(input_size, hidden_size)   # input → hidden
        self.fc_hh = nn.Linear(hidden_size, hidden_size)  # hidden → hidden
        self.fc_out = nn.Linear(hidden_size, output_size)  # hidden → output

    def forward(self, x, h):
        # Input and hidden projections (each delegatable to AXON)
        ih = self.fc_ih(x)
        hh = self.fc_hh(h)
        # Recurrent state update — tanh is NOT TOSA INT-delegatable
        h_new = torch.tanh(ih + hh)
        # Output projection (delegatable to AXON)
        out = self.fc_out(h_new)
        return out, h_new


def main():
    script_dir = Path(__file__).parent
    build_dir = script_dir / "build"
    build_dir.mkdir(exist_ok=True)
    generated_dir = script_dir / "src" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    sdk_path = os.environ.get("SDK_EDGE_AI_PATH", os.path.expanduser("~/sdk-edge-ai"))

    # 1. Create model (no training needed — just demonstrating delegation)
    print("Creating simple RNN step model...")
    model = SimpleRNNStep(input_size=4, hidden_size=8, output_size=2)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Quantize
    print("Quantizing to INT8...")
    from executorch.backends.arm.tosa.specification import TosaSpecification
    from executorch.backends.arm.quantizer import (
        TOSAQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    example_input = (torch.randn(1, 4), torch.randn(1, 8))
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")

    exported = torch.export.export(model, example_input, strict=False)
    captured = exported.module()

    # Strip _guards_fn nodes
    guard_nodes = [
        n for n in captured.graph.nodes
        if n.op == "call_module" and "_guards" in str(n.target)
    ]
    for n in guard_nodes:
        n.replace_all_uses_with(None)
        captured.graph.erase_node(n)
    for name in list(captured._modules.keys()):
        if "_guards" in name:
            delattr(captured, name)
    captured.graph.lint()
    captured.recompile()

    # Quantize only Linear layers — tanh stays in fp32 on CPU
    quantizer = TOSAQuantizer(tosa_spec)
    quantizer.set_module_type(
        nn.Linear, get_symmetric_quantization_config(is_per_channel=False)
    )
    prepared = prepare_pt2e(captured, quantizer)
    # Calibrate
    for _ in range(50):
        prepared(torch.randn(1, 4), torch.randn(1, 8))
    quantized = convert_pt2e(prepared)
    re_exported = torch.export.export(quantized, example_input, strict=False)

    # 3. Partition to AXON and export
    print("Exporting with AXON backend...")
    from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

    compile_spec = AxonCompileSpec(
        sdk_edge_ai_path=sdk_path,
        model_name="rnn_step",
        axon_generated_dir=str(generated_dir),
    )
    partitioner = AxonPartitioner(compile_spec)

    edge = to_edge_transform_and_lower(
        re_exported,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    pte_path = build_dir / "simple_rnn.pte"
    edge.to_executorch().save(str(pte_path))
    print(f"  .pte: {pte_path} ({pte_path.stat().st_size} bytes)")

    # 4. Generate C header
    model_pte_h = script_dir / "src" / "model_pte.h"
    pte_bytes = pte_path.read_bytes()
    with open(model_pte_h, "w") as f:
        f.write("/* Auto-generated from simple_rnn.pte */\n")
        f.write("#include <stdint.h>\n\n")
        f.write("static const uint8_t model_pte[] __attribute__((aligned(16))) = {\n")
        for i, b in enumerate(pte_bytes):
            if i % 16 == 0:
                f.write("  ")
            f.write(f"0x{b:02x},")
            if i % 16 == 15:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"static const uint32_t model_pte_len = {len(pte_bytes)};\n")
    print(f"  C header: {model_pte_h}")

    # List generated AXON headers
    headers = sorted(generated_dir.glob("*.h"))
    subgraph_headers = [h for h in headers
                        if h.name.startswith("axon_subgraph_")
                        and h.name != "axon_subgraphs_table.h"]
    print(f"\n  Generated {len(subgraph_headers)} AXON subgraph(s):")
    for h in subgraph_headers:
        print(f"    {h.name}")
    print(f"  Lookup table: axon_subgraphs_table.h")

    if len(subgraph_headers) >= 2:
        print(f"\n  Multi-subgraph delegation: {len(subgraph_headers)} separate command buffers")
        print("  (The recurrent tanh between FC layers splits them into separate subgraphs.)")
    else:
        print(f"\n  {len(subgraph_headers)} subgraph(s) generated.")

    print("\nDone. Rebuild firmware to embed the model.")


if __name__ == "__main__":
    main()
