#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a model that adds a scalar constant: forward(x) = x + 1.0.

This exercises the scalar-EValue path through aten::add.Scalar (vs
aten::add.Tensor used by the simple test). The scalar `1.0` is encoded
as a Double EValue in the program, not as a constant tensor — so this
hits a different code path than tensor constants.

Output: /tmp/native_scalar.pte and /tmp/native_scalar.ref.
"""

import os
import sys

import torch
from executorch.backends.native.partitioner import NativePartitioner

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export


class ScalarAddModel(torch.nn.Module):
    """forward(x) = x + 1.0 — exercises aten::add.Scalar path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


def main() -> int:
    out_path = os.environ.get("NATIVE_SCALAR_PTE_PATH", "/tmp/native_scalar.pte")
    ref_path = os.environ.get("NATIVE_SCALAR_REF_PATH", "/tmp/native_scalar.ref")

    model = ScalarAddModel().eval()

    example_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    print("=== Exporting ScalarAddModel (delegated to NativeBackend) ===")
    with torch.no_grad():
        eager_out = model(example_input)
    print(f"  Input values:  {example_input.flatten().tolist()}")
    print(f"  Eager output:  {eager_out.flatten().tolist()}")

    ep = export(model, (example_input,))
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
        for i, op in enumerate(plan.operators):
            ovl = op.overload if op.overload else ""
            full = f"{op.name}.{ovl}" if ovl else op.name
            print(f"  op[{i}]: {full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
