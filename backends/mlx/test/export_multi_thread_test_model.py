#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a test model for the multi-threaded inference test.

The model exercises multiple ops and a mutable buffer (KV cache),
producing deterministic outputs that the C++ test can verify.

Model behavior (accumulation via KV cache):
    forward(x, input_pos):
        x: [1, 1, 1, dim]  (input tensor)
        input_pos: [1]      (cache write position, always 0)

        z = relu(x * 2 + 1)                        # always 3.0 with ones input
        old_k = cache.k_cache[:, :, :1, :]          # read old cache at pos 0
        new_val = z + old_k                          # accumulate: 3 + old
        k_cache, v_cache = cache.update(input_pos, new_val, new_val)
        return k_cache[:, :, :1, :] + v_cache[:, :, :1, :]

With all-ones input and input_pos=[0], calling forward N times:
    Call 1: old=0, new_val=3, cache=3.  Output = 3 + 3 = 6.0
    Call 2: old=3, new_val=6, cache=6.  Output = 6 + 6 = 12.0
    Call N: Output = 6.0 * N

The C++ test can verify: output == 6.0 * call_number (all elements).

Usage:
    python export_multi_thread_test_model.py /tmp/multi_thread_test_model.pte
"""

import argparse

import torch
import torch.nn as nn

from executorch.backends.mlx.examples.cache import ETKVCache
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig


class MultiOpCacheModel(nn.Module):
    """
    A model with multiple ops and a mutable KV cache buffer that accumulates.

    Each forward() call:
      1. Computes z = relu(x * 2 + 1)          — mul, add, relu (= 3.0 with ones)
      2. Reads old cache value at pos 0         — old_k
      3. Accumulates: new_val = z + old_k       — add
      4. Writes new_val to both k and v caches  — mutable buffer via kv_cache_update
      5. Returns k_cache + v_cache at pos 0     — sum of both cache slices

    With ones input, output = 6.0 * call_number (all elements).
    """

    def __init__(self, dim=4, max_len=8):
        super().__init__()
        self.cache = ETKVCache(
            max_batch_size=1,
            max_context_length=max_len,
            n_heads=1,
            head_dim=dim,
            enable_dynamic_shape=True,
        )

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        z = torch.relu(x * 2.0 + 1.0)
        old_k = self.cache.k_cache[:, :, :1, :]
        new_val = z + old_k
        k_cache, v_cache = self.cache.update(input_pos, new_val, new_val)
        return k_cache[:, :, :1, :] + v_cache[:, :, :1, :]


def export_model(output_path: str, dim=4, max_len=8):
    model = MultiOpCacheModel(dim=dim, max_len=max_len)
    example_inputs = (
        torch.randn(1, 1, 1, dim),  # x: [B, H, S, D]
        torch.tensor([0], dtype=torch.int64),  # input_pos
    )

    with torch.no_grad():
        exported = torch.export.export(model, example_inputs)
        exported = exported.run_decompositions({})

    et_program = to_edge_transform_and_lower(exported, partitioner=[MLXPartitioner()])
    et_program = et_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"Exported model to {output_path}")

    # Verify accumulation pattern
    model_ref = MultiOpCacheModel(dim=dim, max_len=max_len)
    x = torch.ones(1, 1, 1, dim)
    input_pos = torch.tensor([0], dtype=torch.int64)
    print(f"Reference (ones input, dim={dim}, max_len={max_len}):")
    for i in range(1, 4):
        result = model_ref(x, input_pos)
        expected = 6.0 * i
        actual = result[0, 0, 0, 0].item()
        status = "OK" if abs(actual - expected) < 1e-6 else "FAIL"
        print(f"  Call {i}: output={actual:.1f}, expected={expected:.1f} [{status}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="/tmp/multi_thread_test_model.pte",
        help="Output .pte path (default: /tmp/multi_thread_test_model.pte)",
    )
    args = parser.parse_args()
    export_model(args.output)


if __name__ == "__main__":
    main()
