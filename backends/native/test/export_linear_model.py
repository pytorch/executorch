#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a small model with several constant tensors. Goal: exercise
upload_constants end-to-end on the existing runtime op set (add/mul,
the same ops the simple test uses).

The model is `y = x * scale + bias1 + bias2 * 2.0` over (1, 4) float
tensors, where `scale`, `bias1`, `bias2` are register_buffer
constants. Three constants land in the program's named-data store and
must be uploaded to the engine that consumes them.

Output: /tmp/native_linear.pte and /tmp/native_linear.ref (raw float32
reference output for the C++ test).

(Naming preserved as "linear" for shell-script compatibility; this is
not actually nn.Linear since neither the CPU nor Metal runtime has a
linear kernel today.)
"""

import os
import sys

import torch
from executorch.backends.native.partitioner import NativePartitioner

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export


class ConstantyModel(torch.nn.Module):
    """Model with three frozen tensor constants.

    forward(x) = x * scale + bias1 + bias2 * 2.0

    Three constants (scale, bias1, bias2) flow through the program as
    constant data; AOT bakes them into the .pte and the runtime must
    upload them via Engine::upload_constants before the first execute.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor([2.0, 3.0, 4.0, 5.0]), persistent=False
        )
        self.register_buffer(
            "bias1", torch.tensor([1.0, 2.0, 3.0, 4.0]), persistent=False
        )
        self.register_buffer(
            "bias2", torch.tensor([0.5, 0.25, 0.125, 0.0625]), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias1 + self.bias2


def main() -> int:
    out_path = os.environ.get("NATIVE_LINEAR_PTE_PATH", "/tmp/native_linear.pte")
    ref_path = os.environ.get("NATIVE_LINEAR_REF_PATH", "/tmp/native_linear.ref")

    model = ConstantyModel().eval()

    # Deterministic input the C++ side reproduces.
    example_input = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    print("=== Exporting ConstantyModel (delegated to NativeBackend) ===")
    with torch.no_grad():
        eager_out = model(example_input)
    print(f"  Input values:  {example_input.flatten().tolist()}")
    print(f"  scale:         {model.scale.tolist()}")
    print(f"  bias1:         {model.bias1.tolist()}")
    print(f"  bias2:         {model.bias2.tolist()}")
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

    # Reference output as raw little-endian float32 for the C++ test.
    ref_bytes = eager_out.detach().contiguous().to(torch.float32).numpy().tobytes()
    with open(ref_path, "wb") as f:
        f.write(ref_bytes)
    print(f"  Saved reference output {ref_path} ({len(ref_bytes)} bytes)")

    # Surface the operator list so we can see what landed in the delegate.
    prog = et._emitter_output.program
    for ep_i, plan in enumerate(prog.execution_plan):
        for i, op in enumerate(plan.operators):
            ovl = op.overload if op.overload else ""
            full = f"{op.name}.{ovl}" if ovl else op.name
            print(f"  op[{i}]: {full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
