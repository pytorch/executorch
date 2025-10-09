# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export


class TestCudaExport(unittest.TestCase):
    """Test CUDA export functionality for various operations using to_edge_transform_and_lower."""

    def setUp(self):
        """Set up test environment."""
        # Skip tests if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def _export_to_cuda_with_lower(
        self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
    ) -> None:
        """Helper method to export a module to CUDA backend using to_edge_transform_and_lower."""
        # Export the model
        exported_program = export(module, inputs, strict=True)

        # Create partitioner and compile specs
        partitioner = CudaPartitioner(
            [CudaBackend.generate_method_name_compile_spec("forward")]
        )

        # Use to_edge_transform_and_lower for complete pipeline
        edge_program_manager = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )

        # Verify that the pipeline succeeded
        self.assertIsNotNone(edge_program_manager)
        self.assertTrue(hasattr(edge_program_manager, "exported_program"))

        # Verify that the final exported program contains delegated calls
        exported_program = edge_program_manager.exported_program()
        has_delegate_call = False
        for node in exported_program.graph.nodes:
            if node.op == "call_function" and "executorch_call_delegate" in str(
                node.target
            ):
                has_delegate_call = True
                break

        self.assertTrue(
            has_delegate_call, "No delegate calls found in final exported program"
        )

        return edge_program_manager

    def test_simple_add(self):
        """Test CUDA export for simple element-wise addition."""

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        module = AddModule()
        module.eval()
        inputs = (torch.randn(3, 4), torch.randn(3, 4))

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Simple add operation export failed")

    def test_conv2d(self):
        """Test CUDA export for 2D convolution."""

        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        module = Conv2dModule()
        module.eval()
        inputs = (torch.randn(1, 3, 32, 32),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Conv2d operation export failed")

    def test_linear(self):
        """Test CUDA export for linear layer."""

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        module = LinearModule()
        module.eval()
        inputs = (torch.randn(8, 128),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Linear operation export failed")

    def test_resnet_block(self):
        """Test CUDA export for a ResNet-style block."""

        class ResNetBlock(torch.nn.Module):
            def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
                # Use eval mode to avoid batch norm mutations during export
                self.bn1 = torch.nn.BatchNorm2d(out_channels)
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = torch.nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                self.bn2 = torch.nn.BatchNorm2d(out_channels)

                # Shortcut connection
                self.shortcut = torch.nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        torch.nn.BatchNorm2d(out_channels),
                    )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                identity = self.shortcut(x)

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                out += identity
                out = self.relu(out)

                return out

        module = ResNetBlock(64, 64)
        # Set module to eval mode to avoid batch norm running statistics mutations
        module.eval()
        inputs = (torch.randn(1, 64, 32, 32),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "ResNet block export failed")

    def test_multi_operation_module(self):
        """Test CUDA export for a module with multiple operations."""

        class MultiOpModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.linear = torch.nn.Linear(32, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                return x

        module = MultiOpModule()
        module.eval()
        inputs = (torch.randn(2, 3, 16, 16),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(
            edge_program_manager, "Multi-operation module export failed"
        )

    def test_activation_functions(self):
        """Test CUDA export for various activation functions."""

        class ActivationModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Test multiple activation functions
                x1 = torch.relu(x)
                x2 = torch.sigmoid(x)
                x3 = torch.tanh(x)
                return x1 + x2 + x3

        module = ActivationModule()
        module.eval()
        inputs = (torch.randn(4, 8),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Activation functions export failed")

    def test_mathematical_operations(self):
        """Test CUDA export for mathematical operations."""

        class MathOpsModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # Test various mathematical operations
                add_result = x + y
                mul_result = x * y
                sub_result = x - y
                div_result = x / (y + 1e-8)  # Add epsilon to avoid division by zero
                return add_result + mul_result + sub_result + div_result

        module = MathOpsModule()
        module.eval()
        inputs = (torch.randn(4, 4), torch.randn(4, 4))

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(
            edge_program_manager, "Mathematical operations export failed"
        )
