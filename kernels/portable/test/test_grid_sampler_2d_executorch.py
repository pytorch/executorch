#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test grid_sampler_2d by exporting to ExecuTorch and comparing with PyTorch.
"""

import itertools
import sys
import unittest

import torch
import torch.nn as nn
from executorch.exir import to_edge
from executorch.runtime import Runtime
from torch.export import export


class GridSampleModule(nn.Module):
    """Wrapper module for grid_sample operation."""

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            input,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class GridSampler2DExecutorchTest(unittest.TestCase):
    """Test ExecuTorch grid_sampler_2d implementation."""

    def run_executorch_test(
        self,
        input_tensor: torch.Tensor,
        grid: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> None:
        """Export to ExecuTorch and compare with PyTorch reference."""

        # Create module
        model = GridSampleModule(mode, padding_mode, align_corners)
        model.eval()

        # PyTorch reference
        with torch.no_grad():
            pytorch_output = model(input_tensor, grid)

        try:
            # Export to ExecuTorch
            example_inputs = (input_tensor, grid)

            # Export the model
            exported_program = export(model, example_inputs)

            # Convert to edge IR
            edge_program = to_edge(exported_program)

            # Get ExecuTorch program
            executorch_program = edge_program.to_executorch()

            # Run through ExecuTorch
            runtime = Runtime.get()
            fwd_method = runtime.load_program(executorch_program.buffer).load_method(
                "forward"
            )
            if fwd_method is None:
                self.fail("Failed to load forward method")
            executorch_output = fwd_method.execute((input_tensor, grid))[0]

            # Compare results
            self.assertTrue(
                executorch_output.shape == pytorch_output.shape,
                msg=f"Shape mismatch: ET={executorch_output.shape} vs PT={pytorch_output.shape}",
            )

            if not torch.allclose(
                executorch_output, pytorch_output, atol=atol, rtol=rtol
            ):
                max_diff = (executorch_output - pytorch_output).abs().max().item()
                mean_diff = (executorch_output - pytorch_output).abs().mean().item()
                self.fail(
                    f"\nMode: {mode}, Padding: {padding_mode}, Align: {align_corners}\n"
                    f"Max difference: {max_diff:.6e}\n"
                    f"Mean difference: {mean_diff:.6e}\n"
                    f"Tolerance (atol): {atol:.6e}\n"
                    f"ExecuTorch output:\n{executorch_output}\n"
                    f"PyTorch output:\n{pytorch_output}\n"
                )

        except Exception as e:
            self.fail(
                f"Failed to export or run model:\n"
                f"Mode: {mode}, Padding: {padding_mode}, Align: {align_corners}\n"
                f"Error: {str(e)}"
            )

    def test_all_mode_combinations(self):
        """Test all combinations of interpolation modes, padding modes, and align_corners."""
        print("\n" + "=" * 70)
        print("Testing all mode combinations")
        print("=" * 70)

        modes = ["bilinear", "nearest", "bicubic"]
        padding_modes = ["zeros", "border", "reflection"]
        align_corners_options = [True, False]

        # Test parameters
        batch_size = 2
        channels = 3
        height_in = 5
        width_in = 5
        height_out = 4
        width_out = 4

        test_count = 0
        for mode, padding, align in itertools.product(
            modes, padding_modes, align_corners_options
        ):
            with self.subTest(mode=mode, padding=padding, align=align):
                # Create random input
                input_tensor = torch.randn(
                    batch_size, channels, height_in, width_in, dtype=torch.float32
                )

                # Create grid with some values in [-1, 1] range
                grid = torch.randn(
                    batch_size, height_out, width_out, 2, dtype=torch.float32
                )
                grid = torch.clamp(grid, -1.2, 1.2)  # Include some out-of-bounds

                # Bicubic may have slightly larger numerical errors
                atol = 1e-4 if mode == "bicubic" else 1e-5

                self.run_executorch_test(
                    input_tensor, grid, mode, padding, align, atol=atol
                )
                test_count += 1
                print(f"  ✓ {mode}/{padding}/align={align}")

        print(f"✓ Passed {test_count} mode combination tests")

    def test_batch_sizes(self):
        """Test various batch sizes."""
        print("\n" + "=" * 70)
        print("Testing various batch sizes")
        print("=" * 70)

        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(batch_size, 3, 6, 6, dtype=torch.float32)
                grid = torch.randn(batch_size, 4, 4, 2, dtype=torch.float32)
                grid = torch.clamp(grid, -1, 1)

                self.run_executorch_test(input_tensor, grid, "bilinear", "zeros", False)
                print(f"  ✓ batch_size={batch_size}")

        print(f"✓ Passed {len(batch_sizes)} batch size tests")

    def test_input_sizes(self):
        """Test various input and output sizes."""
        print("\n" + "=" * 70)
        print("Testing various input/output sizes")
        print("=" * 70)

        test_cases = [
            # (H_in, W_in, H_out, W_out)
            (4, 4, 4, 4),  # Same size
            (8, 8, 4, 4),  # Downsampling
            (4, 4, 8, 8),  # Upsampling
            (10, 5, 7, 3),  # Non-square, different aspect ratios
        ]

        for h_in, w_in, h_out, w_out in test_cases:
            with self.subTest(h_in=h_in, w_in=w_in, h_out=h_out, w_out=w_out):
                input_tensor = torch.randn(1, 2, h_in, w_in, dtype=torch.float32)
                grid = torch.randn(1, h_out, w_out, 2, dtype=torch.float32)
                grid = torch.clamp(grid, -1, 1)

                self.run_executorch_test(input_tensor, grid, "bilinear", "zeros", False)
                print(f"  ✓ {h_in}x{w_in} -> {h_out}x{w_out}")

        print(f"✓ Passed {len(test_cases)} size variation tests")

    def test_identity_grid(self):
        """Test with identity grid (should return approximately same as input)."""
        print("\n" + "=" * 70)
        print("Testing identity grid")
        print("=" * 70)

        input_tensor = torch.randn(1, 3, 4, 4, dtype=torch.float32)

        # Create identity grid
        y = torch.linspace(-1, 1, 4)
        x = torch.linspace(-1, 1, 4)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        for mode in ["bilinear", "nearest", "bicubic"]:
            with self.subTest(mode=mode):
                atol = 1e-4 if mode == "bicubic" else 1e-5
                self.run_executorch_test(
                    input_tensor, grid, mode, "zeros", True, atol=atol
                )
                print(f"  ✓ {mode}")

        print("✓ Passed identity grid tests")

    def test_corner_coordinates(self):
        """Test sampling at corner coordinates with different align_corners settings."""
        print("\n" + "=" * 70)
        print("Testing corner coordinates")
        print("=" * 70)

        input_tensor = torch.randn(1, 1, 8, 8, dtype=torch.float32)

        # Grid sampling at corners
        grid = torch.tensor(
            [
                [
                    [[-1.0, -1.0], [-1.0, 1.0]],
                    [[1.0, -1.0], [1.0, 1.0]],
                ]
            ],
            dtype=torch.float32,
        )

        for align_corners in [True, False]:
            for mode in ["bilinear", "nearest"]:
                with self.subTest(align_corners=align_corners, mode=mode):
                    self.run_executorch_test(
                        input_tensor, grid, mode, "zeros", align_corners
                    )
                    print(f"  ✓ {mode}/align={align_corners}")

        print("✓ Passed corner coordinate tests")

    def test_out_of_bounds(self):
        """Test behavior with out-of-bounds coordinates for different padding modes."""
        print("\n" + "=" * 70)
        print("Testing out-of-bounds coordinates")
        print("=" * 70)

        input_tensor = torch.randn(1, 2, 5, 5, dtype=torch.float32)

        # Grid with out-of-bounds coordinates
        grid = torch.tensor(
            [
                [
                    [[-1.5, -1.5], [-0.5, -0.5], [0.5, 0.5], [1.5, 1.5]],
                    [[-1.0, 1.5], [0.0, 0.0], [1.0, -1.5], [2.0, 2.0]],
                ]
            ],
            dtype=torch.float32,
        )

        for padding_mode in ["zeros", "border", "reflection"]:
            for mode in ["bilinear", "nearest"]:
                with self.subTest(padding_mode=padding_mode, mode=mode):
                    self.run_executorch_test(
                        input_tensor, grid, mode, padding_mode, False
                    )
                    print(f"  ✓ {mode}/{padding_mode}")

        print("✓ Passed out-of-bounds tests")

    def test_single_channel(self):
        """Test with single channel input."""
        print("\n" + "=" * 70)
        print("Testing single channel input")
        print("=" * 70)

        input_tensor = torch.randn(1, 1, 6, 6, dtype=torch.float32)
        grid = torch.randn(1, 4, 4, 2, dtype=torch.float32)
        grid = torch.clamp(grid, -1, 1)

        for mode in ["bilinear", "nearest", "bicubic"]:
            with self.subTest(mode=mode):
                atol = 1e-4 if mode == "bicubic" else 1e-5
                self.run_executorch_test(
                    input_tensor, grid, mode, "zeros", False, atol=atol
                )
                print(f"  ✓ {mode}")

        print("✓ Passed single channel tests")

    def test_different_dtypes(self):
        """Test with different data types (float16, bfloat16)."""
        print("\n" + "=" * 70)
        print("Testing different dtypes")
        print("=" * 70)

        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                input_tensor = torch.randn(1, 2, 4, 4, dtype=dtype)
                grid = torch.randn(1, 3, 3, 2, dtype=dtype)
                grid = torch.clamp(grid, -1, 1)

                # Use larger tolerance for float16/bfloat16
                atol = 1e-2 if dtype == torch.bfloat16 else 5e-3

                self.run_executorch_test(
                    input_tensor, grid, "bilinear", "zeros", False, atol=atol
                )
                print(f"  ✓ {dtype}")

        print("✓ Passed dtype tests")

    def test_very_small_inputs(self):
        """Test with very small input sizes."""
        print("\n" + "=" * 70)
        print("Testing very small inputs")
        print("=" * 70)

        test_cases = [
            # (H_in, W_in, H_out, W_out, description)
            (1, 1, 1, 1, "1x1 input, 1x1 output"),
            (1, 1, 2, 2, "1x1 input, 2x2 output"),
            (2, 2, 1, 1, "2x2 input, 1x1 output"),
            (2, 2, 2, 2, "2x2 input, 2x2 output"),
            (2, 2, 3, 3, "2x2 input, 3x3 output"),
            (3, 3, 1, 1, "3x3 input, single pixel output"),
        ]

        for h_in, w_in, h_out, w_out, desc in test_cases:
            with self.subTest(desc=desc):
                input_tensor = torch.randn(1, 2, h_in, w_in, dtype=torch.float32)
                grid = torch.randn(1, h_out, w_out, 2, dtype=torch.float32)
                grid = torch.clamp(grid, -1, 1)

                self.run_executorch_test(input_tensor, grid, "bilinear", "zeros", False)
                print(f"  ✓ {desc}")

        print("✓ Passed very small input tests")

    def test_exact_boundary_coordinates(self):
        """Test with grid coordinates exactly at boundaries."""
        print("\n" + "=" * 70)
        print("Testing exact boundary coordinates")
        print("=" * 70)

        input_tensor = torch.randn(1, 2, 5, 5, dtype=torch.float32)

        # Test grid with exact boundary values
        grids = [
            # All corners
            torch.tensor(
                [[[[-1.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [1.0, 1.0]]]],
                dtype=torch.float32,
            ),
            # Center
            torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float32),
            # Edges
            torch.tensor(
                [[[[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        ]

        for i, grid in enumerate(grids):
            for mode in ["bilinear", "nearest", "bicubic"]:
                for align_corners in [True, False]:
                    with self.subTest(grid=i, mode=mode, align_corners=align_corners):
                        atol = 1e-4 if mode == "bicubic" else 1e-5
                        self.run_executorch_test(
                            input_tensor, grid, mode, "zeros", align_corners, atol=atol
                        )
                        print(f"  ✓ grid {i}/{mode}/align={align_corners}")

        print("✓ Passed exact boundary coordinate tests")

    def test_out_of_bounds_values_in_grid(self):
        """Test with out of bounds values in grid."""
        print("\n" + "=" * 70)
        print("Testing special values in grid")
        print("=" * 70)

        input_tensor = torch.randn(1, 2, 4, 4, dtype=torch.float32)

        test_cases = [
            # (grid, description)
            (
                torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]], dtype=torch.float32),
                "Very large coordinates (far out of bounds)",
            ),
            (
                torch.tensor(
                    [[[[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]]]],
                    dtype=torch.float32,
                ),
                "Moderately out of bounds coordinates",
            ),
        ]

        for grid, desc in test_cases:
            with self.subTest(desc=desc):
                # Test with zeros padding (most common for out-of-bounds)
                self.run_executorch_test(input_tensor, grid, "bilinear", "zeros", False)
                print(f"  ✓ {desc}")

        print("✓ Passed special value tests")


def main():
    """Run the tests."""
    print("\n" + "=" * 70)
    print("ExecuTorch grid_sampler_2d Test Suite")
    print("Testing via model export and ExecuTorch runtime")
    print("=" * 70)

    # Run tests with verbose output
    suite = unittest.TestLoader().loadTestsFromTestCase(GridSampler2DExecutorchTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
