#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a model that exercises ALL THREE compute providers in a single
plan. Goal: show off the router actually splitting work across
fake_accel + metal + cpu.

Forward pass: cos((x + 1.0) @ w)

  - aten::add.Tensor       → routed to FAKE_ACCEL
                             (it's first in the candidate order and
                             claims aten::add)
  - aten::mm  (matmul)     → routed to METAL
                             (fake_accel doesn't claim mm; Metal does)
  - aten::cos.out          → routed to CPU
                             (only cpu has cos in its op set)

The model needs compute_unit="fake_accel|metal|cpu" at runtime.
"auto" (= "cpu|metal") would skip fake_accel and route the add to
metal, missing the demo.

Output: /tmp/native_tri.pte and /tmp/native_tri.ref.
"""

import os
import sys

import torch
from executorch.backends.native.partitioner import NativePartitioner

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export


class TriProviderModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        a = x + 1.0
        b = torch.mm(a, w)  # explicit aten::mm (Metal registers under that name)
        c = torch.cos(b)
        return c


def main() -> int:
    out_path = os.environ.get("NATIVE_TRI_PTE_PATH", "/tmp/native_tri.pte")
    ref_path = os.environ.get("NATIVE_TRI_REF_PATH", "/tmp/native_tri.ref")

    model = TriProviderModel().eval()
    # Deterministic inputs so the C++ side reproduces them exactly.
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])  # (1, 4)
    w = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
    )  # (4, 3)

    print("=== Exporting TriProviderModel (delegated to NativeBackend) ===")
    with torch.no_grad():
        eager_out = model(x, w)
    print(f"  x: {x.flatten().tolist()}")
    print(f"  w (4x3): {w.flatten().tolist()}")
    print(f"  Eager output: {eager_out.flatten().tolist()}")

    ep = export(model, (x, w))
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[NativePartitioner()],
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
    et = edge.to_executorch()

    with open(out_path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {out_path} ({len(et.buffer)} bytes)")

    ref_bytes = eager_out.detach().contiguous().to(torch.float32).numpy().tobytes()
    with open(ref_path, "wb") as f:
        f.write(ref_bytes)
    print(f"  Saved reference output {ref_path} ({len(ref_bytes)} bytes)")

    prog = et._emitter_output.program
    for ep_i, plan in enumerate(prog.execution_plan):
        print(f"  ExecutionPlan ops:")
        for i, op in enumerate(plan.operators):
            ovl = op.overload if op.overload else ""
            full = f"{op.name}.{ovl}" if ovl else op.name
            print(f"    op[{i}]: {full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
