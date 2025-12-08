# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for bitwise shift operations in ExecuTorch.

This test validates that the bitwise left/right shift operators work correctly
by creating simple models that use these operations and running inference.
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


class BitwiseLeftShiftModule(torch.nn.Module):
    """Module that uses bitwise left shift with tensor operand."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_left_shift(x, y)


class BitwiseRightShiftModule(torch.nn.Module):
    """Module that uses bitwise right shift with tensor operand."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_right_shift(x, y)


class BitwiseLeftShiftScalarModule(torch.nn.Module):
    """Module that uses bitwise left shift with scalar operand."""

    def __init__(self, shift_amount: int = 2):
        super().__init__()
        self.shift_amount = shift_amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_left_shift(x, self.shift_amount)


class BitwiseRightShiftScalarModule(torch.nn.Module):
    """Module that uses bitwise right shift with scalar operand."""

    def __init__(self, shift_amount: int = 2):
        super().__init__()
        self.shift_amount = shift_amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_right_shift(x, self.shift_amount)


def export_and_generate_pte(model, example_inputs, output_path):
    """Export a model and generate a .pte file."""
    exported_program = torch.export.export(model, example_inputs)
    edge_program_manager = to_edge_transform_and_lower(
        exported_program,
        partitioner=None,
        compile_config=EdgeCompileConfig(
            _core_aten_ops_exception_list=[
                torch.ops.aten.bitwise_left_shift.Tensor,
                torch.ops.aten.bitwise_left_shift.Tensor_Scalar,
                torch.ops.aten.bitwise_right_shift.Tensor,
                torch.ops.aten.bitwise_right_shift.Tensor_Scalar,
            ]
        ),
    )
    executorch_program_manager = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(executorch_program_manager, str(output_path))


class TestBitwiseShiftOperators(unittest.TestCase):
    """Test bitwise shift operators in ExecuTorch."""

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
        torch.testing.assert_close(outputs[0], expected)
        return outputs[0]

    # ==========================================================================
    # Core tests: one per operator signature
    # ==========================================================================

    def test_left_shift_tensor_tensor(self):
        """Test bitwise_left_shift.Tensor_out: (Tensor, Tensor) -> Tensor."""
        model = BitwiseLeftShiftModule()
        x = torch.tensor([[1, 2, 4, 8], [16, 32, 64, 128]], dtype=torch.int32)
        y = torch.tensor([[1, 2, 1, 2], [3, 2, 1, 0]], dtype=torch.int32)
        self._run_and_compare(model, (x, y), "left_shift_tensor_tensor.pte")

    def test_left_shift_tensor_scalar(self):
        """Test bitwise_left_shift.Tensor_Scalar_out: (Tensor, Scalar) -> Tensor."""
        model = BitwiseLeftShiftScalarModule(shift_amount=3)
        x = torch.tensor([[1, 2, 4, 8], [16, 32, 64, 128]], dtype=torch.int32)
        self._run_and_compare(model, (x,), "left_shift_tensor_scalar.pte")

    def test_right_shift_tensor_tensor(self):
        """Test bitwise_right_shift.Tensor_out: (Tensor, Tensor) -> Tensor."""
        model = BitwiseRightShiftModule()
        x = torch.tensor([[8, 16, 32, 64], [128, 256, 512, 1024]], dtype=torch.int32)
        y = torch.tensor([[1, 2, 1, 2], [3, 4, 5, 6]], dtype=torch.int32)
        self._run_and_compare(model, (x, y), "right_shift_tensor_tensor.pte")

    def test_right_shift_tensor_scalar(self):
        """Test bitwise_right_shift.Tensor_Scalar_out: (Tensor, Scalar) -> Tensor."""
        model = BitwiseRightShiftScalarModule(shift_amount=3)
        x = torch.tensor([[8, 16, 32, 64], [128, 256, 512, 1024]], dtype=torch.int32)
        self._run_and_compare(model, (x,), "right_shift_tensor_scalar.pte")

    # ==========================================================================
    # Edge cases
    # ==========================================================================

    def test_shift_by_zero(self):
        """Test that shifting by zero returns original values."""
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        # Test Tensor_Scalar variants (scalar shift amount)
        self._run_and_compare(
            BitwiseLeftShiftScalarModule(shift_amount=0),
            (x,),
            "left_shift_scalar_zero.pte",
        )
        self._run_and_compare(
            BitwiseRightShiftScalarModule(shift_amount=0),
            (x,),
            "right_shift_scalar_zero.pte",
        )
        # Test Tensor variants (tensor shift amount)
        zero_shift = torch.zeros_like(x)
        self._run_and_compare(
            BitwiseLeftShiftModule(), (x, zero_shift), "left_shift_tensor_zero.pte"
        )
        self._run_and_compare(
            BitwiseRightShiftModule(), (x, zero_shift), "right_shift_tensor_zero.pte"
        )

    def test_different_dtypes(self):
        """Test bitwise shift with different integer dtypes."""
        for dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            with self.subTest(dtype=dtype):
                # Use smaller values for int8 to avoid overflow
                if dtype == torch.int8:
                    x = torch.tensor([1, 2, 4, 8], dtype=dtype)
                    y = torch.tensor([1, 1, 1, 1], dtype=dtype)
                    shift = 1
                else:
                    x = torch.tensor([8, 16, 32, 64], dtype=dtype)
                    y = torch.tensor([1, 2, 1, 3], dtype=dtype)
                    shift = 2
                # Test Tensor variants (tensor shift amount)
                self._run_and_compare(
                    BitwiseLeftShiftModule(), (x, y), f"left_shift_tensor_{dtype}.pte"
                )
                self._run_and_compare(
                    BitwiseRightShiftModule(), (x, y), f"right_shift_tensor_{dtype}.pte"
                )
                # Test Tensor_Scalar variants (scalar shift amount)
                self._run_and_compare(
                    BitwiseLeftShiftScalarModule(shift_amount=shift),
                    (x,),
                    f"left_shift_scalar_{dtype}.pte",
                )
                self._run_and_compare(
                    BitwiseRightShiftScalarModule(shift_amount=shift),
                    (x,),
                    f"right_shift_scalar_{dtype}.pte",
                )


if __name__ == "__main__":
    unittest.main()
