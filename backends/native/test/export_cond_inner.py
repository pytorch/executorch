#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export the cond model as a RAW flatbuffer Program (no PTE wrapper, no
partitioner) so the C++ bypass test can feed it directly to
NativeBackend::init() as if it were the delegate's `processed` payload.

The output is exactly the shape NativeBackend's preprocess produces as
its inner program — but for a model the partitioner can't claim today
(see TODO in native_partitioner.py for HOPs). This bypasses the
partitioner-side wrap issue and exercises the runtime path directly.

Output: /tmp/native_cond_inner.fbb (flatbuffer Program bytes only)
"""

import os
import sys

import torch

from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir._serialize._flatbuffer_program import _program_to_flatbuffer
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
    out_path = os.environ.get("NATIVE_COND_INNER_PATH", "/tmp/native_cond_inner.fbb")

    print("=== Exporting CondModel to raw flatbuffer Program ===")
    ep = export(
        CondModel().eval(),
        (torch.tensor(True), torch.ones(2, 3)),
    )
    edge = to_edge(ep, compile_config=EdgeCompileConfig(_skip_dim_order=True))
    et = edge.to_executorch()
    # _program_to_flatbuffer keeps program.constant_buffer in place (no
    # constant_segment extraction), matching what NativeBackend's preprocess
    # does for its delegate inner program.
    fb = _program_to_flatbuffer(et._emitter_output.program)
    with open(out_path, "wb") as f:
        f.write(bytes(fb.data))
    print(f"  Saved {out_path} ({len(fb.data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
