#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a tiny model to a .pte file for the C++ test_model runner.

The model is `c = a + b` over (1, 4) float tensors. It exports through
ExecuTorch's standard pipeline and lowers to the v2 NativeBackend via
NativePartitioner, exercising the AOT path
(NativePartitioner + NativeBackend.preprocess) end-to-end. Add
decomposes to `aten::add.Tensor.out` which is dispatched via the
default CPU handler in NativeBackend's CpuOps.

(Earlier model: TinyLinear. Linear is in the partitioner's preserve list
because Metal has a dedicated kernel; on CPU-only it has no handler
since `aten::linear.default` isn't an out-variant. `add` works on both
paths and is the simplest end-to-end smoke test.)

Output: /tmp/native_simple.pte
"""

import os
import sys

import torch
from torch.export import export

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.backends.native.partitioner import NativePartitioner


class TinyAdd(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def main() -> int:
    out_path = os.environ.get("NATIVE_SIMPLE_PTE_PATH", "/tmp/native_simple.pte")

    model = TinyAdd().eval()
    example = (torch.randn(1, 4), torch.randn(1, 4))

    print(f"=== Exporting TinyAdd (delegated to NativeBackend) ===")
    with torch.no_grad():
        eager_out = model(*example)
    print(f"  Eager output shape: {tuple(eager_out.shape)}")
    print(f"  Eager output values: {eager_out.flatten().tolist()}")

    ep = export(model, example)
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[NativePartitioner()],
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
    et = edge.to_executorch()

    with open(out_path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {out_path} ({len(et.buffer)} bytes)")

    # Surface the operator list so we can see what landed in the delegate
    # (vs at the top level).
    prog = et._emitter_output.program
    for ep_i, plan in enumerate(prog.execution_plan):
        for i, op in enumerate(plan.operators):
            ovl = op.overload if op.overload else ""
            full = f"{op.name}.{ovl}" if ovl else op.name
            print(f"  op[{i}]: {full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
