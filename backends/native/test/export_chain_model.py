#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a model with a tensor add followed by a chain of unary ops:

  forward(x, y) = tanh(exp(sin(cos(x + y))))

Goals:
  - Exercise multiple ops in a single delegate segment.
  - Trigger ET's Reinplace pass: each unary op feeds the next, so its
    input is single-use and can be turned into in-place. The CPU
    runtime's `aten::cos_`, `aten::sin_`, `aten::exp_`, `aten::tanh_`
    handlers (registered in CpuOps.cpp) get exercised.
  - Surface the post-route partition log: NativeBackend prints which
    runtime each segment lands on.
"""

import os
import sys

import torch
from torch.export import export

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.backends.native.partitioner import NativePartitioner


class ChainModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = x + y
        z = torch.cos(z)
        z = torch.sin(z)
        z = torch.exp(z)
        z = torch.tanh(z)
        return z


def main() -> int:
    out_path = os.environ.get(
        "NATIVE_CHAIN_PTE_PATH", "/tmp/native_chain.pte"
    )
    ref_path = os.environ.get(
        "NATIVE_CHAIN_REF_PATH", "/tmp/native_chain.ref"
    )

    model = ChainModel().eval()
    a = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    b = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    print("=== Exporting ChainModel (delegated to NativeBackend) ===")
    with torch.no_grad():
        eager_out = model(a, b)
    print(f"  Inputs: a={a.flatten().tolist()} b={b.flatten().tolist()}")
    print(f"  Eager output: {eager_out.flatten().tolist()}")

    ep = export(model, (a, b))
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[NativePartitioner()],
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
    et = edge.to_executorch()

    with open(out_path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {out_path} ({len(et.buffer)} bytes)")

    ref_bytes = (
        eager_out.detach().contiguous().to(torch.float32).numpy().tobytes()
    )
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
