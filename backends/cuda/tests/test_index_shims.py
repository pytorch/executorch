# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mask indexing and index_put in the CUDA backend's index shims.

A mask index makes a result size depend on the data, so Inductor leaves it,
and an index_put with a mask, to the backend's own index shims. These tests
lower real programs, run them through the Python runtime, and compare with
eager PyTorch.
"""

import os
import tempfile
import unittest
from typing import Sequence, Tuple

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipIf(
    torch.version.hip is not None,
    "loads through the Python runtime, which the ROCm build does not include",
)
class TestIndexShims(unittest.TestCase):
    def _run(
        self, model: torch.nn.Module, cases: Sequence[Tuple[torch.Tensor, ...]]
    ) -> None:
        """Lower model for cases[0], then run every case and compare with eager."""
        spec = CudaBackend.generate_method_name_compile_spec("forward")
        lowered = to_edge_transform_and_lower(
            export(model, cases[0]),
            partitioner=[CudaPartitioner([spec])],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        # Everything runs in the delegate, the index ops included.
        self.assertEqual(
            {
                n.target.__name__
                for n in lowered.exported_program().graph.nodes
                if n.op == "call_function"
            },
            {"executorch_call_delegate", "getitem"},
        )
        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                lowered.to_executorch().write_to_file(f)
            method = Runtime.get().load_program(path).load_method("forward")
            for inputs in cases:
                outputs = method.execute(list(inputs))
                for output, expected in zip(outputs, model(*inputs), strict=True):
                    torch.testing.assert_close(output.cpu(), expected)

    def test_mask_index_and_index_put(self):
        """index_put with accumulate goes to the shim too: Inductor has no atomic
        add for int64."""

        class MaskIndex(torch.nn.Module):
            def forward(self, x, mask, column, idx, counts):
                out = x.clone()
                out[mask] = x[mask] * 2
                # The mask and column index dims 0 and 2, which are not adjacent.
                picked = x[mask, :, column].sum(0)
                # A transposed input gives a result that is not row-major.
                rows = x.transpose(1, 2)[mask] * 3
                ones = torch.ones_like(idx)
                counts = counts.index_put((idx,), ones, accumulate=True)
                return out, picked, rows, counts

        x = torch.randn(4, 5, 6)
        column = torch.tensor([2])
        idx = torch.tensor([0, 3, 3, 1, 3])
        # Nonzero counts check that repeated writes add to the old values.
        counts = torch.arange(8) * 5
        masks = (
            torch.tensor([True, False, True, True]),
            torch.zeros(4, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
        )
        self._run(MaskIndex(), [(x, mask, column, idx, counts) for mask in masks])

    def test_two_dim_mask(self):
        class TwoDimMask(torch.nn.Module):
            def forward(self, x, mask):
                return (x[mask] * 2,)

        x = torch.randn(4, 5, 6)
        masks = (torch.rand(4, 5) > 0.5, torch.zeros(4, 5, dtype=torch.bool))
        self._run(TwoDimMask(), [(x, mask) for mask in masks])

    def test_index_put_with_repeated_indices(self):
        """Deterministic mode sends half, bfloat16 and int32-index writes to the shim."""

        class Put(torch.nn.Module):
            def forward(self, x, idx, values):
                summed = x.index_put((idx,), values, accumulate=True)
                # Without accumulate, the last of the repeated writes wins.
                written = x.index_put((idx.int(),), values)
                return summed, written

        idx = torch.tensor([0, 3, 3, 1, 3])
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    x = torch.arange(8, dtype=dtype)
                    values = torch.arange(1, 6, dtype=dtype)
                    self._run(Put(), [(x, idx, values)])
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
