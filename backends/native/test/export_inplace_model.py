#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a model that uses explicit in-place ops at the source level.

  forward(x):
      y = x.clone()       # clone so we don't mutate caller buffer
      y.add_(1.0)         # aten::add_.Scalar
      y.mul_(2.0)         # aten::mul_.Scalar
      y.relu_()           # aten::relu_
      return y

For input x = [-2, -1, 0, 1, 2]:
  after add_(1.0):  [-1, 0, 1, 2, 3]
  after mul_(2.0):  [-2, 0, 2, 4, 6]
  after relu_():    [ 0, 0, 2, 4, 6]

Output: /tmp/native_inplace.pte
"""

import os
import sys

import torch
from torch.export import export

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.backends.native.partitioner import NativePartitioner


class InplaceModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        y.add_(1.0)
        y.mul_(2.0)
        y.relu_()
        return y


def main() -> int:
    out_path = os.environ.get(
        "NATIVE_INPLACE_PTE_PATH", "/tmp/native_inplace.pte"
    )

    model = InplaceModel().eval()
    example = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("=== Exporting InplaceModel (delegated to NativeBackend) ===")
    with torch.no_grad():
        eager_out = model(example.clone())
    print(f"  Input:  {example.tolist()}")
    print(f"  Eager:  {eager_out.tolist()}")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
