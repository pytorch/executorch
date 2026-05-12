#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a stateful model with a mutable buffer accumulator.

Module:
    self.acc = zeros(2, 3)        # register_buffer
    forward(x): self.acc.add_(x)
                return self.acc.clone()

Calling forward(ones(2,3)) repeatedly should accumulate:
    call 1 → ones
    call 2 → twos
    call 3 → threes

Output: /tmp/native_stateful.pte
"""

import os
import sys

import torch
from torch.export import export

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.backends.native.partitioner import NativePartitioner


class StatefulAdd(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("acc", torch.zeros(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.acc.add_(x)
        return self.acc.clone()


def main() -> int:
    out_path = os.environ.get(
        "NATIVE_STATEFUL_PTE_PATH", "/tmp/native_stateful.pte"
    )

    model = StatefulAdd().eval()
    example = torch.ones(2, 3)

    print("=== Exporting StatefulAdd (delegated to NativeBackend) ===")
    print("  Initial acc: zeros(2, 3)")
    print("  Per-call input: ones(2, 3)")
    print("  Expected outputs: ones, twos, threes (cumulative)")

    ep = export(model, (example,))
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[NativePartitioner()],
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
    et = edge.to_executorch()

    with open(out_path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {out_path} ({len(et.buffer)} bytes)")

    prog = et._emitter_output.program
    for plan in prog.execution_plan:
        for i, op in enumerate(plan.operators):
            ovl = op.overload if op.overload else ""
            full = f"{op.name}.{ovl}" if ovl else op.name
            print(f"  op[{i}]: {full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
