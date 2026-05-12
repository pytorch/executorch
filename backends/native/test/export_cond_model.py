#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a model that uses torch.cond for control flow.

Module:
    forward(pred: Bool, x: Float) -> Float
        if pred: x + x   else: x * x

Tested twice in the C++ driver:
  - pred=True,  x=ones(2,3) -> twos
  - pred=False, x=ones(2,3) -> ones (1*1=1)

Output: /tmp/native_cond.pte

Note: torch.cond export creates HOPs (higher-order ops) that go through
a different path than ordinary partitioning. If this script or the
runtime fails for HOPs, that's a known forward-looking feature.
"""

import os
import sys

import torch
from executorch.backends.native.partitioner import NativePartitioner

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export


class CondModel(torch.nn.Module):
    def forward(self, pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(
            pred,
            lambda x: x + x,
            lambda x: x * x,
            (x,),
        )


def main() -> int:
    out_path = os.environ.get("NATIVE_COND_PTE_PATH", "/tmp/native_cond.pte")

    model = CondModel().eval()
    pred = torch.tensor(True)
    x = torch.ones(2, 3)

    print("=== Exporting CondModel (delegated to NativeBackend) ===")
    print("  forward(pred, x): if pred then x+x else x*x")

    try:
        ep = export(model, (pred, x))
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[NativePartitioner()],
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        )
        et = edge.to_executorch()
    except Exception as e:
        print(f"  Export FAILED: {type(e).__name__}: {e}")
        return 1

    with open(out_path, "wb") as f:
        f.write(et.buffer)
    print(f"  Saved {out_path} ({len(et.buffer)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
