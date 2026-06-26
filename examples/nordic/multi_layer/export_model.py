#!/usr/bin/env python3
# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Export a multi-layer classifier for the AXON NPU.

Demonstrates multi-subgraph delegation: each Linear layer becomes
its own AXON-compiled subgraph with a separate command buffer header.
The delegate lookup table maps subgraph names to compiled models.

Model: 8 inputs → 32 → 16 → 4 outputs (3 FC layers, 3 AXON subgraphs)

Usage:
    SDK_EDGE_AI_PATH=~/sdk-edge-ai ./run_export.sh
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn


class MultiLayerClassifier(nn.Module):
    """Two-branch classifier with independent AXON subgraphs.

    Branch A: fc_a (8 -> 16)
    Branch B: fc_b (8 -> 16)
    Merge:    element-wise multiply (breaks the delegation chain)
    Head:     fc_head (16 -> 4)

    The element-wise multiply between the branches forces the
    partitioner to create separate AXON subgraphs for each branch
    and the head, demonstrating multi-subgraph delegation.
    """

    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(8, 16)
        self.fc_b = nn.Linear(8, 16)
        self.fc_head = nn.Linear(16, 4)

    def forward(self, x):
        a = torch.relu(self.fc_a(x))
        b = torch.relu(self.fc_b(x))
        merged = a * b
        return self.fc_head(merged)


def main():
    script_dir = Path(__file__).parent
    build_dir = script_dir / "build"
    build_dir.mkdir(exist_ok=True)
    generated_dir = script_dir / "src" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    sdk_path = os.environ.get("SDK_EDGE_AI_PATH", os.path.expanduser("~/sdk-edge-ai"))

    # 1. Train on a simple classification task (XOR-like)
    print("Training multi-layer classifier...")
    model = MultiLayerClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Generate training data: classify 8-dim input into 4 classes
    torch.manual_seed(42)
    x_train = torch.randn(500, 8)
    # Labels based on which quadrant the first two dims fall into
    y_train = ((x_train[:, 0] > 0).long() * 2 + (x_train[:, 1] > 0).long())

    model.train()
    for epoch in range(500):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = (model(x_train).argmax(dim=1) == y_train).float().mean()
    print(f"  Final loss: {loss.item():.4f}, accuracy: {accuracy:.1%}")

    # 2. Quantize
    print("Quantizing to INT8...")
    from executorch.backends.arm.tosa.specification import TosaSpecification
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    model.eval()
    example_input = (torch.randn(1, 8),)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")

    exported = torch.export.export(model, example_input, strict=False)
    captured = exported.module()

    # Strip torch 2.11 _guards_fn nodes
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

    quantizer = EthosUQuantizer(tosa_spec).set_global(
        get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = prepare_pt2e(captured, quantizer)
    # Calibrate
    for _ in range(50):
        prepared(torch.randn(1, 8))
    quantized = convert_pt2e(prepared)
    re_exported = torch.export.export(quantized, example_input, strict=False)

    # 3. Partition to AXON and export
    print("Exporting with AXON backend...")
    from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

    compile_spec = AxonCompileSpec(
        sdk_edge_ai_path=sdk_path,
        model_name="multi_layer",
        axon_generated_dir=str(generated_dir),
    )
    partitioner = AxonPartitioner(compile_spec)

    edge = to_edge_transform_and_lower(
        re_exported,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    pte_path = build_dir / "multi_layer.pte"
    edge.to_executorch().save(str(pte_path))
    print(f"  .pte: {pte_path} ({pte_path.stat().st_size} bytes)")

    # 4. Generate C header from .pte (16-byte aligned)
    model_pte_h = script_dir / "src" / "model_pte.h"
    pte_bytes = pte_path.read_bytes()
    with open(model_pte_h, "w") as f:
        f.write("/* Auto-generated from multi_layer.pte */\n")
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
    subgraph_headers = [h for h in headers if h.name.startswith("axon_subgraph_") and h.name != "axon_subgraphs_table.h"]
    print(f"\n  Generated {len(subgraph_headers)} AXON subgraph(s):")
    for h in subgraph_headers:
        print(f"    {h.name}")
    print(f"  Lookup table: axon_subgraphs_table.h")

    print(f"\n  AXON compiler chained all layers into {len(subgraph_headers)} command buffer(s).")
    print("  (Models with non-delegatable ops between layers produce multiple subgraphs.)")

    print("\nDone. Rebuild firmware to embed the model.")


if __name__ == "__main__":
    main()
