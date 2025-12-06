# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
import unittest

import torch


class GridSampler2dTest(unittest.TestCase):
    def run_grid_sampler_test(
        self,
        inp: torch.Tensor,
        grid: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        atol: float = 1e-5,
    ) -> None:
        """Test grid_sampler_2d against PyTorch's reference implementation."""
        # PyTorch reference
        aten_result = torch.nn.functional.grid_sample(
            inp,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        # Convert mode strings to integers for et_test
        mode_map = {"bilinear": 0, "nearest": 1, "bicubic": 2}
        padding_map = {"zeros": 0, "border": 1, "reflection": 2}

        # ExecuTorch implementation
        et_result = torch.zeros_like(aten_result)
        et_result = torch.ops.et_test.grid_sampler_2d(
            inp,
            grid,
            interpolation_mode=mode_map[mode],
            padding_mode=padding_map[padding_mode],
            align_corners=align_corners,
            out=et_result,
        )

        self.assertTrue(
            torch.allclose(et_result, aten_result, atol=atol, rtol=1e-5),
            msg=f"Mode: {mode}, Padding: {padding_mode}, Align: {align_corners}\n"
            f"ET: {et_result}\n"
            f"ATen: {aten_result}\n"
            f"Error: {(et_result.to(torch.float) - aten_result.to(torch.float)).abs().max()}",
        )

    def test_grid_sampler_2d_all_modes_f32(self):
        """Test all combinations of interpolation, padding, and align_corners."""
        N = [1, 2]
        C = [1, 3]
        H_IN = [4, 8]
        W_IN = [4, 8]
        H_OUT = [3, 6]
        W_OUT = [3, 6]
        MODES = ["bilinear", "nearest", "bicubic"]
        PADDING_MODES = ["zeros", "border", "reflection"]
        ALIGN_CORNERS = [True, False]

        for (
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            mode,
            padding_mode,
            align_corners,
        ) in itertools.product(
            N, C, H_IN, W_IN, H_OUT, W_OUT, MODES, PADDING_MODES, ALIGN_CORNERS
        ):
            # Create input tensor
            input_tensor = torch.randn(n, c, h_in, w_in, dtype=torch.float32)

            # Create grid with coordinates in [-1, 1]
            grid = torch.randn(n, h_out, w_out, 2, dtype=torch.float32)
            # Normalize grid to [-1, 1] range
            grid = torch.clamp(grid, -2, 2)

            self.run_grid_sampler_test(
                input_tensor,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                atol=1e-4,  # Slightly relaxed tolerance for bicubic
            )

    def test_grid_sampler_2d_bilinear_specific_cases(self):
        """Test bilinear mode with specific edge cases."""
        # Test with identity grid (should return same as input)
        input_tensor = torch.randn(1, 3, 4, 4)
        y = torch.linspace(-1, 1, 4)
        x = torch.linspace(-1, 1, 4)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        self.run_grid_sampler_test(
            input_tensor,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def test_grid_sampler_2d_nearest_specific_cases(self):
        """Test nearest mode with specific patterns."""
        # Create a checkerboard pattern
        input_tensor = torch.zeros(1, 1, 4, 4)
        input_tensor[0, 0, ::2, ::2] = 1.0
        input_tensor[0, 0, 1::2, 1::2] = 1.0

        # Sample at grid points
        y = torch.linspace(-1, 1, 6)
        x = torch.linspace(-1, 1, 6)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        self.run_grid_sampler_test(
            input_tensor,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

    def test_grid_sampler_2d_padding_modes(self):
        """Test different padding modes with out-of-bounds coordinates."""
        input_tensor = torch.randn(1, 2, 5, 5)

        # Create grid with some out-of-bounds coordinates
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
            for align_corners in [True, False]:
                self.run_grid_sampler_test(
                    input_tensor,
                    grid,
                    mode="bilinear",
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )

    def test_grid_sampler_2d_bicubic_smoothness(self):
        """Test bicubic interpolation for smooth gradients."""
        # Create a smooth gradient
        input_tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

        # Create a fine grid for upsampling
        y = torch.linspace(-1, 1, 7)
        x = torch.linspace(-1, 1, 7)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        self.run_grid_sampler_test(
            input_tensor,
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=True,
            atol=1e-4,
        )

    def test_grid_sampler_2d_align_corners_comparison(self):
        """Compare align_corners=True vs False."""
        input_tensor = torch.randn(1, 1, 8, 8)

        # Create grid at corner positions
        grid = torch.tensor(
            [
                [
                    [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
                ]
            ],
            dtype=torch.float32,
        )

        for mode in ["bilinear", "nearest", "bicubic"]:
            # Test with align_corners=True
            self.run_grid_sampler_test(
                input_tensor,
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )

            # Test with align_corners=False
            self.run_grid_sampler_test(
                input_tensor,
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=False,
            )

    def test_grid_sampler_2d_batch_processing(self):
        """Test with multiple batches."""
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 6, 6)
            grid = torch.randn(batch_size, 4, 4, 2)
            grid = torch.clamp(grid, -1.5, 1.5)

            self.run_grid_sampler_test(
                input_tensor,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )


if __name__ == "__main__":
    unittest.main()
