# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for bucketize operations in ExecuTorch.

This test validates that the bucketize operator work correctly
by creating simple models that use the operation and running inference.
"""

import tempfile
import unittest
from pathlib import Path

import torch
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from executorch.runtime import Runtime


class BucketizeModule(torch.nn.Module):
    """Module that uses bucketize"""

    def __init__(self, out_int32: bool, right: bool):
        super().__init__()
        self.out_int32 = out_int32
        self.right = right

    def forward(self, x, bounds: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(x, bounds, out_int32=self.out_int32, right=self.right)


def export_and_generate_pte(model, example_inputs, output_path):
    """Export a model and generate a .pte file."""
    exported_program = torch.export.export(model, example_inputs)
    edge_program_manager = to_edge_transform_and_lower(
        exported_program,
        partitioner=None,
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[
                torch.ops.aten.bucketize.Tensor,
                torch.ops.aten.bucketize.Scalar,
            ]
        ),
    )
    executorch_program_manager = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(executorch_program_manager, str(output_path))


class TestBucketizeOperator(unittest.TestCase):
    """Test bucketize operator in ExecuTorch."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_and_compare(self, model, inputs, pte_name):
        """Helper to export, run, and compare outputs."""
        model.eval()
        expected = model(*inputs)

        pte_path = self.temp_path / pte_name
        export_and_generate_pte(model, inputs, pte_path)

        runtime = Runtime.get()
        method = runtime.load_program(pte_path).load_method("forward")
        outputs = method.execute(list(inputs))

        self.assertEqual(len(outputs), 1)
        print(outputs[0])
        print(expected)
        torch.testing.assert_close(outputs[0], expected)
        return outputs[0]

    # ==========================================================================
    # Core tests: one per operator signature
    # ==========================================================================

    def test_bucketize_tensor_out_int64(self):
        """Test bucketize.Tensor_out: (Tensor, Tensor, bool, bool) -> Tensor."""
        model = BucketizeModule(False, False)
        x = torch.tensor([[1, 4, 6, 8]], dtype=torch.float)
        bounds = torch.tensor([0, 3, 5, 7, 9], dtype=torch.float)
        self._run_and_compare(model, (x, bounds), "test_bucketize_tensor_out_int64.pte")

    def test_bucketize_tensor_out_int32(self):
        """Test bucketize.Tensor_out: (Tensor, Tensor, bool, bool) -> Tensor."""
        model = BucketizeModule(True, False)
        x = torch.tensor([[1, 4, 6, 8]], dtype=torch.float)
        bounds = torch.tensor([0, 3, 5, 7, 9], dtype=torch.float)
        self._run_and_compare(model, (x, bounds), "test_bucketize_tensor_out_int32.pte")

    def test_bucketize_tensor_right(self):
        """Test bucketize.Tensor_out: (Tensor, Tensor, bool, bool) -> Tensor."""
        model = BucketizeModule(False, True)
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float)
        bounds = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        self._run_and_compare(model, (x, bounds), "test_bucketize_tensor_right.pte")

    def test_bucketize_tensor_left(self):
        """Test bucketize.Tensor_out: (Tensor, Tensor, bool, bool) -> Tensor."""
        model = BucketizeModule(False, False)
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float)
        bounds = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        self._run_and_compare(model, (x, bounds), "test_bucketize_tensor_left.pte")

    def test_bucketize_scalar_out_int64(self):
        """Test bucketize.Tensor_out: (Scalar, Tensor, bool, bool) -> Tensor."""
        model = BucketizeModule(False, False)
        x = 1
        bounds = torch.tensor([0, 3, 5, 7, 9], dtype=torch.float)
        self._run_and_compare(model, (x, bounds), "test_bucketize_scalar_out_int64.pte")

    def test_bucketize_scalar_out_int32(self):
        """Test bucketize.Tensor_out: (Scalar, Tensor, bool, bool) -> Tensor."""
        model = BucketizeModule(False, False)
        x = 1
        bounds = torch.tensor([0, 3, 5, 7, 9], dtype=torch.float)
        self._run_and_compare(model, (x, bounds), "test_bucketize_scalar_out_int32.pte")


if __name__ == "__main__":
    unittest.main()
