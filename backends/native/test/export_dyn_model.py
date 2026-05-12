#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a model with a dynamic batch dim, exercised at varying batch
sizes per call against the same loaded delegate.

Model:  forward(x) = (x + 1.0) * 2.0
  shapes: x is (B, 4) where B in [1, 8]; traced at B=4.

For input x of shape (B, 4) with all-ones values:
  expected output = (1 + 1) * 2 = 4.0 everywhere.

Output: /tmp/native_dyn.pte
"""

import os
import sys

import torch
from torch.export import export, Dim

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.backends.native.partitioner import NativePartitioner


class DynModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 2.0


def main() -> int:
    out_path = os.environ.get("NATIVE_DYN_PTE_PATH", "/tmp/native_dyn.pte")

    model = DynModel().eval()
    example = torch.ones(4, 4)  # trace at batch=4

    print("=== Exporting DynModel (delegated to NativeBackend) ===")
    print("  Trace shape: (4, 4); dynamic batch dim B in [1, 8]")

    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = {"x": {0: batch}}

    ep = export(model, (example,), dynamic_shapes=dynamic_shapes)
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[NativePartitioner()],
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    )
    et = edge.to_executorch()

    with open(out_path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {out_path} ({len(et.buffer)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
