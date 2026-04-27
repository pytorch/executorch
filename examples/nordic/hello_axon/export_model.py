#!/usr/bin/env python3
# Copyright (c) 2026 iote.ai
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Export a simple FC model for the AXON NPU.

Trains a 3-layer FC network on sin(x), quantizes to INT8,
partitions to AXON, and exports as .pte + generated headers.

Usage:
    # After running setup_export_env.sh:
    PYTHONHOME= SDK_EDGE_AI_PATH=~/sdk-edge-ai uv run python export_model.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn


class SineModel(nn.Module):
    """3-layer FC: 1 -> 16 -> 16 -> 1. Approximates sin(x)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    script_dir = Path(__file__).parent
    build_dir = script_dir / "build"
    build_dir.mkdir(exist_ok=True)
    generated_dir = script_dir / "src" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    sdk_path = os.environ.get("SDK_EDGE_AI_PATH", os.path.expanduser("~/sdk-edge-ai"))

    # 1. Train
    print("Training sine model...")
    model = SineModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x_train = torch.linspace(0, 2 * math.pi, 1000).unsqueeze(1)
    y_train = torch.sin(x_train)

    model.train()
    for epoch in range(1000):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  Final loss: {loss.item():.6f}")

    # 2. Quantize
    print("Quantizing to INT8...")
    from executorch.backends.arm.tosa.specification import TosaSpecification
    from executorch.backends.arm.quantizer import (
        EthosUQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    model.eval()
    example_input = (torch.randn(1, 1),)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")

    exported = torch.export.export(model, example_input, strict=False)
    captured = exported.module()

    # Torch 2.11 quirk: export().module() inserts _guards_fn call_module
    # nodes that the ExecuTorch pass manager doesn't handle. Strip them.
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
    for _ in range(100):
        prepared(torch.rand(1, 1) * 2 * math.pi)
    quantized = convert_pt2e(prepared)
    re_exported = torch.export.export(quantized, example_input, strict=False)

    # 3. Partition to AXON and export
    print("Exporting with AXON backend...")
    from executorch.backends.nordic.axon import AxonCompileSpec, AxonPartitioner
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

    compile_spec = AxonCompileSpec(
        sdk_edge_ai_path=sdk_path,
        model_name="hello_axon",
        axon_generated_dir=str(generated_dir),
    )
    partitioner = AxonPartitioner(compile_spec)

    edge = to_edge_transform_and_lower(
        re_exported,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    pte_path = build_dir / "hello_axon.pte"
    edge.to_executorch().save(str(pte_path))
    print(f"  .pte: {pte_path} ({pte_path.stat().st_size} bytes)")

    # 4. Generate C header from .pte (16-byte aligned for ExecuTorch)
    model_pte_h = script_dir / "src" / "model_pte.h"
    pte_bytes = pte_path.read_bytes()
    with open(model_pte_h, "w") as f:
        f.write("/* Auto-generated from hello_axon.pte */\n")
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
    headers = list(generated_dir.glob("*.h"))
    print(f"  Generated {len(headers)} header(s) in {generated_dir}/")
    for h in sorted(headers):
        print(f"    {h.name}")

    print("\nDone. Rebuild firmware to embed the model.")


if __name__ == "__main__":
    main()
