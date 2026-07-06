#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate test data (delegate flatbuffer blobs) for GraphTypes C++ tests."""

import pathlib

import torch
import torch.nn as nn

from executorch.backends.native.partitioner import NativePartitioner
from executorch.exir import to_edge_transform_and_lower


def _generate_linear_blob() -> bytes:
    model = nn.Linear(4, 4)
    ep = torch.export.export(model, (torch.randn(1, 4),))
    edge = to_edge_transform_and_lower(ep, partitioner=[NativePartitioner()])
    et = edge.to_executorch()
    delegates = et.executorch_program.backend_delegate_data
    assert len(delegates) == 1
    return bytes(delegates[0].data)


class _DiamondModel(nn.Module):
    """x -> add(x,x) -> a; a -> mul(a,2) -> b; a -> add(a,1) -> c; add(b,c) -> out.

    'a' has 2 users (mul and add), creating an interesting use-def graph.
    """

    def forward(self, x):
        a = x + x
        b = a * 2
        c = a + 1
        return b + c


def _generate_diamond_blob() -> bytes:
    model = _DiamondModel()
    ep = torch.export.export(model, (torch.randn(1, 4),))
    edge = to_edge_transform_and_lower(ep, partitioner=[NativePartitioner()])
    et = edge.to_executorch()
    delegates = et.executorch_program.backend_delegate_data
    assert len(delegates) == 1
    return bytes(delegates[0].data)


class _KVCacheModel(nn.Module):
    """HF-style KV cache with index_copy_ (lowered to index_put).

    k_cache is a mutable buffer [1, max_seq=8, head_dim=4].
    forward updates the cache at position `pos` and returns a reduction.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, 8, 4))

    def forward(self, k_new, pos):
        self.k_cache.index_copy_(1, pos, k_new)
        return self.k_cache.sum(dim=1)


def _generate_kv_cache_blob() -> bytes:
    model = _KVCacheModel()
    k_new = torch.randn(1, 1, 4)
    pos = torch.tensor([0])
    ep = torch.export.export(model, (k_new, pos), strict=False)
    edge = to_edge_transform_and_lower(ep, partitioner=[NativePartitioner()])
    et = edge.to_executorch()
    delegates = et.executorch_program.backend_delegate_data
    assert len(delegates) == 1
    return bytes(delegates[0].data)


def main():
    out_dir = pathlib.Path(__file__).parent / "testdata"
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = _generate_linear_blob()
    out_path = out_dir / "linear_4x4.bin"
    out_path.write_bytes(blob)
    print(f"Wrote {len(blob)} bytes to {out_path}")

    blob = _generate_diamond_blob()
    out_path = out_dir / "diamond.bin"
    out_path.write_bytes(blob)
    print(f"Wrote {len(blob)} bytes to {out_path}")

    blob = _generate_kv_cache_blob()
    out_path = out_dir / "kv_cache.bin"
    out_path.write_bytes(blob)
    print(f"Wrote {len(blob)} bytes to {out_path}")


if __name__ == "__main__":
    main()
