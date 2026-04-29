#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export

try:
    import executorch.exir as exir
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    _MLX_RUNTIME_OK = sys.platform == "darwin"
except (AttributeError, ImportError, OSError):
    _MLX_RUNTIME_OK = False


class _Unfold1DModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.unfold(x.unsqueeze(-1), kernel_size=(3, 1), stride=(1, 1))


class _SeparatedAdvancedIndexModel(nn.Module):
    def forward(
        self, x: torch.Tensor, idx0: torch.Tensor, idx2: torch.Tensor
    ) -> torch.Tensor:
        return x[idx0, :, idx2]


def _run_with_mlx(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    exported = export(module.eval(), inputs, strict=True)
    edge = exir.to_edge_transform_and_lower(
        exported,
        partitioner=[MLXPartitioner()],
        compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
    )
    lowered = edge.to_executorch(
        config=exir.ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    runtime_module = _load_for_executorch_from_buffer(lowered.buffer)
    return runtime_module.forward(list(inputs))[0]


@unittest.skipUnless(_MLX_RUNTIME_OK, "MLX runtime tests require macOS + pybindings")
class TestMLXRuntimeOps(unittest.TestCase):
    def test_unfold_1d_preserves_channel_major_patch_order(self):
        x = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)
        module = _Unfold1DModel()

        with torch.no_grad():
            expected = module(x)
            actual = _run_with_mlx(module, (x,))

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_separated_advanced_indices_keep_broadcast_dims_front(self):
        x = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
        idx0 = torch.tensor([[0], [1]], dtype=torch.long)
        idx2 = torch.tensor([[0, 2, 4]], dtype=torch.long)
        module = _SeparatedAdvancedIndexModel()

        with torch.no_grad():
            expected = module(x, idx0, idx2)
            actual = _run_with_mlx(module, (x, idx0, idx2))

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
