# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test sort CUDA shim for AOTI export.

The sort shim (sort.cu) provides aoti_torch_cuda_sort_stable, a thrust-based
fallback for aten::sort.stable that Inductor emits when it can't natively lower
sort. This is needed for ops like argsort that decompose to sort_stable.

Usage:
  python -m pytest backends/cuda/tests/test_sort_shim.py -v
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export


class SortModel(nn.Module):
    """Model that uses sort (via argsort) for export testing."""

    def forward(self, x):
        # argsort decomposes to sort_stable in Inductor
        return x.argsort(dim=-1)


class SortStableModel(nn.Module):
    """Model that uses torch.sort directly."""

    def forward(self, x):
        values, indices = torch.sort(x, dim=-1, stable=True)
        return values, indices


class TestSortShim(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_argsort_export(self):
        """argsort exports and produces .pte via AOTI with sort shim."""
        model = SortModel().eval()
        x = torch.randn(4, 8, dtype=torch.float32, device="cuda")

        with torch.no_grad():
            ep = export(model, (x,), strict=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [CudaBackend.generate_method_name_compile_spec("forward")]
            et_prog = to_edge_transform_and_lower(
                ep,
                partitioner=[CudaPartitioner(specs)],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False, _skip_dim_order=True
                ),
            )
            et_program = et_prog.to_executorch(
                config=ExecutorchBackendConfig(
                    extract_delegate_segments=True,
                    do_quant_fusion_and_const_prop=True,
                    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
                ),
            )

            pte_path = os.path.join(tmpdir, "sort_model.pte")
            with open(pte_path, "wb") as f:
                et_program.write_to_file(f)

            self.assertTrue(os.path.exists(pte_path))
            self.assertGreater(os.path.getsize(pte_path), 0)

    def test_sort_stable_export(self):
        """torch.sort(stable=True) exports and produces .pte via AOTI with sort shim."""
        model = SortStableModel().eval()
        x = torch.randn(4, 8, dtype=torch.float32, device="cuda")

        with torch.no_grad():
            ep = export(model, (x,), strict=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [CudaBackend.generate_method_name_compile_spec("forward")]
            et_prog = to_edge_transform_and_lower(
                ep,
                partitioner=[CudaPartitioner(specs)],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False, _skip_dim_order=True
                ),
            )
            et_program = et_prog.to_executorch(
                config=ExecutorchBackendConfig(
                    extract_delegate_segments=True,
                    do_quant_fusion_and_const_prop=True,
                    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
                ),
            )

            pte_path = os.path.join(tmpdir, "sort_stable_model.pte")
            with open(pte_path, "wb") as f:
                et_program.write_to_file(f)

            self.assertTrue(os.path.exists(pte_path))
            self.assertGreater(os.path.getsize(pte_path), 0)

    def test_sort_fallback_registered(self):
        """sort_stable is registered as a supported fallback kernel."""
        fallbacks = CudaBackend.get_supported_fallback_kernels()
        self.assertIn("at::_ops::sort_stable::call", fallbacks)


if __name__ == "__main__":
    unittest.main()
