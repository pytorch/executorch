# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import unittest
from typing import Tuple

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend

from executorch.exir import EdgeProgramManager, to_edge
from torch.export import Dim, export, ExportedProgram

ctypes.CDLL("libvulkan.so.1")


from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten


class TestBackends(unittest.TestCase):
    def assert_outputs_equal(
        self, model_output, ref_output, atol=1e-03, rtol=1e-03, first_output_only=False
    ):
        """
        Helper testing function that asserts that the model output and the reference output
        are equal with some tolerance. Due to numerical differences between eager mode and
        the Vulkan's backend, we relax the detal such that default absolute
        tolerance is 1e-3. and default relative tolerance is 1e-3.
        """

        # Compare the result from executor and eager mode direclty
        if isinstance(ref_output, tuple) or isinstance(ref_output, list):
            # Multiple outputs executor always returns tuple, even if there is one output
            self.assertTrue(len(ref_output) == len(model_output))
            if first_output_only:
                self.assertTrue(
                    torch.allclose(model_output[0], ref_output[0], atol=atol, rtol=rtol)
                )
            else:
                for i in range(len(ref_output)):
                    self.assertTrue(
                        torch.allclose(
                            model_output[i], ref_output[i], atol=atol, rtol=rtol
                        )
                    )
        else:
            # If one output, eager returns tensor while executor tuple of size 1
            self.assertTrue(
                torch.allclose(model_output[0], ref_output, atol=atol, rtol=rtol)
            )

    def lower_module_and_test_output(
        self,
        model: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        atol=1e-03,
        rtol=1e-01,
        dynamic_shapes=None,
        test_inputs=None,
        memory_layouts=None,
        first_output_only=False,
    ):
        """
        Helper testing function that takes a torch.nn.Module and lowers it to Vulkan with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """

        def run_test(memory_layout):
            compile_options = {
                "memory_layout_override": memory_layout,
            }
            program: ExportedProgram = export(
                model, sample_inputs, dynamic_shapes=dynamic_shapes
            )
            edge_program: EdgeProgramManager = to_edge(program)

            edge_program = edge_program.to_backend(VulkanPartitioner(compile_options))

            executorch_program = edge_program.to_executorch()

            self.assertEqual(
                executorch_program.executorch_program.execution_plan[0].delegates[0].id,
                VulkanBackend.__name__,
            )

            executorch_module = _load_for_executorch_from_buffer(
                executorch_program.buffer
            )
            inputs_flattened, _ = tree_flatten(sample_inputs)

            model_output = executorch_module.run_method(
                "forward", tuple(inputs_flattened)
            )
            ref_output = model(*sample_inputs)

            self.assert_outputs_equal(
                model_output,
                ref_output,
                atol=atol,
                rtol=rtol,
                first_output_only=first_output_only,
            )

            if test_inputs is not None:
                for test_input in test_inputs:
                    test_inputs_flattened, _ = tree_flatten(test_input)
                    model_output = executorch_module.run_method(
                        "forward", tuple(test_inputs_flattened)
                    )
                    ref_output = model(*test_input)

                    self.assert_outputs_equal(
                        model_output,
                        ref_output,
                        atol=atol,
                        rtol=rtol,
                        first_output_only=first_output_only,
                    )

        memory_layouts_to_test = [
            vk_graph_schema.VkMemoryLayout.TENSOR_WIDTH_PACKED,
            vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED,
        ]

        if memory_layouts is not None:
            memory_layouts_to_test = memory_layouts

        for memory_layout in memory_layouts_to_test:
            run_test(memory_layout)

    def test_vulkan_backend_add(self):
        # This test is the simplest test by manually lowering some submodules, we can use paritioner
        # for auto detecting lowerable parts.
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, w):
                z = x + y
                z = z + x
                z = z + x
                z = z + w
                z = w + z
                z = z + 3  # test scalar broadcasting
                return z

        add_module = AddModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 1), dtype=torch.float32),  # test broadcasting
        )

        self.lower_module_and_test_output(add_module, sample_inputs)

    def test_vulkan_backend_internal_data(self):
        class InternalDataModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.rand(size=(2, 3), dtype=torch.float32)

            def forward(self, x, y):
                z = torch.add(x, y, alpha=2)
                z = torch.add(x, y, alpha=3.14)
                z = z + x
                z = z + self.weight
                return z

        internal_data_module = InternalDataModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(internal_data_module, sample_inputs)

    def test_vulkan_backend_sub(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.sub(x, y, alpha=2)
                z = torch.sub(z, x, alpha=3.14)
                z = z - x
                return z

        sub_module = SubModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(sub_module, sample_inputs)

    def test_vulkan_backend_mul(self):
        class MulModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x * y
                z = z * x
                z = z * x
                return z

        mul_module = MulModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(mul_module, sample_inputs)

    def test_vulkan_backend_div(self):
        class DivModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x / y
                z = z / x
                z = z / x
                return z

        div_module = DivModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(div_module, sample_inputs)

    def test_vulkan_backend_arithmetic(self):
        class ArithmeticModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.rand(size=(2, 3), dtype=torch.float32)

            def forward(self, x, y):
                z = x + y
                z = z - x
                z = z / x
                z = z * self.weight
                return z

        arithmetic_module = ArithmeticModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(arithmetic_module, sample_inputs)

    def test_vulkan_backend_floor_div(self):
        class FloorDivModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x // y
                return z

        floor_div_module = FloorDivModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32) * 10.0,
            torch.rand(size=(2, 3), dtype=torch.float32) + 1.0,
        )

        # absolute tolerance is 1 because of flooring
        self.lower_module_and_test_output(
            floor_div_module, sample_inputs, atol=1.0 + 1e-03
        )

    def test_vulkan_backend_pow(self):
        class PowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.pow(x, y)
                return z

        pow_module = PowModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(pow_module, sample_inputs)

    def lower_clamp_module_and_test_output(self, module):
        batch = Dim("batch", max=8)
        sample_inputs = (torch.randn(8, 16, 96, 92),)

        dynamic_shapes = {"x": {0: batch}}
        test_inputs = [
            (torch.randn(3, 14, 15, 92),),
            (torch.randn(6, 5, 35, 89),),
            (torch.randn(7, 9, 32, 38),),
        ]
        self.lower_module_and_test_output(
            module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
        )

    def test_vulkan_backend_clamp(self):
        class ClampModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.clamp(x, min=-3.14)

        self.lower_clamp_module_and_test_output(ClampModule())

    def test_vulkan_backend_hardtanh(self):
        class HardTanHModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tanh = torch.nn.Hardtanh(min_val=-3.14, max_val=6.28)

            def forward(self, x):
                return self.tanh(x)

        self.lower_clamp_module_and_test_output(HardTanHModule())

    def test_vulkan_backend_relu(self):
        class ReLUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.relu(x)

        self.lower_clamp_module_and_test_output(ReLUModule())

    def test_vulkan_backend_max_pool2d(self):
        class MaxPool2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool = torch.nn.MaxPool2d(
                    kernel_size=(2, 3),
                    stride=(1, 1),
                    padding=0,
                    dilation=1,
                    ceil_mode=False,
                    return_indices=True,
                )

            def forward(self, x):
                return self.max_pool(x)

        max_pool2d_module = MaxPool2dModule()
        sample_inputs = (torch.randn(5, 13, 55, 68),)

        batch = Dim("batch", max=8)
        dynamic_shapes = {"x": {0: batch}}
        test_inputs = [
            (torch.randn(3, 14, 15, 9),),
            (torch.randn(1, 1, 4, 6),),
            (torch.randn(5, 10, 50, 40),),
        ]
        self.lower_module_and_test_output(
            max_pool2d_module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
            first_output_only=True,
        )

    def test_vulkan_backend_abs(self):
        class AbsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.abs(x)

        self.lower_clamp_module_and_test_output(AbsModule())

    def test_vulkan_backend_sigmoid(self):
        class SigmoidModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sigmoid(x)

        self.lower_clamp_module_and_test_output(SigmoidModule())

    def test_vulkan_backend_tanh(self):
        class TanhModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.tanh(x)

        self.lower_clamp_module_and_test_output(TanhModule())

    def test_vulkan_backend_partial(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.offset_1 = torch.rand(size=(2, 10), dtype=torch.float32)
                self.offset_2 = torch.rand(size=(2, 10), dtype=torch.float32)

            def forward(self, x):
                return self.linear(x + self.offset_1) - self.offset_2

        model = SimpleModel()
        sample_inputs = (torch.rand(size=(2, 10), dtype=torch.float32),)

        self.lower_module_and_test_output(model, sample_inputs)

    def test_vulkan_backend_partial_dynamic_shapes(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.branch1 = torch.nn.Sequential(
                    torch.nn.Linear(64, 64), torch.nn.ReLU()
                )
                self.branch2 = torch.nn.Sequential(
                    torch.nn.Linear(128, 64), torch.nn.ReLU()
                )
                self.buffer_1 = torch.ones((1, 64)) * 0.5
                self.buffer_2 = torch.ones((1, 64)) * 1.4

            def forward(self, x1, x2):
                out1 = self.branch1(x1)
                out2 = self.branch2(x2)
                return (out1 + self.buffer_1 + out2) * self.buffer_2

        model = SimpleModel()
        sample_inputs = (torch.randn(32, 64), torch.randn(32, 128))
        batch = Dim("batch", max=32)
        dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

        test_inputs = [
            (torch.randn(15, 64), torch.randn(15, 128)),
            (torch.randn(6, 64), torch.randn(6, 128)),
            (torch.randn(30, 64), torch.randn(30, 128)),
            (torch.randn(20, 64), torch.randn(20, 128)),
            (torch.randn(19, 64), torch.randn(19, 128)),
        ]

        self.lower_module_and_test_output(
            model, sample_inputs, dynamic_shapes=dynamic_shapes, test_inputs=test_inputs
        )

    def test_vulkan_backend_matmul(self):
        class MatMulModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.ones(size=(63, 22), dtype=torch.float32)

            def forward(self, x):
                return torch.matmul(x, self.weight)

        module = MatMulModule()
        sample_inputs = (torch.ones(size=(31, 63), dtype=torch.float32),)

        self.lower_module_and_test_output(module, sample_inputs)

    def test_vulkan_backend_sum_dim_list(self):
        class SumModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.sum(x, (0, -1), keepdim=True)
                x = torch.sum(x, 2, keepdim=False)
                return x

        module = SumModule()
        sample_inputs = (torch.ones(size=(3, 2, 7, 5), dtype=torch.float32),)

        self.lower_module_and_test_output(
            module,
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_conv2d(self):
        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=6,
                    out_channels=8,
                    kernel_size=(3, 3),
                    padding=(2, 3),
                    stride=(1, 2),
                    dilation=1,
                    groups=1,
                    bias=True,
                )

            def forward(self, x):
                return self.conv(x)

        conv2d_module = Conv2dModule()
        sample_inputs = (torch.randn(size=(1, 6, 40, 50), dtype=torch.float32),)

        self.lower_module_and_test_output(
            conv2d_module,
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_conv_transpose2d(self):
        class ConvTranspose2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(
                    in_channels=6,
                    out_channels=8,
                    kernel_size=(3, 3),
                    padding=(2, 3),
                    stride=(1, 2),
                    output_padding=(0, 1),
                    dilation=1,
                    groups=1,
                    bias=True,
                )

            def forward(self, x):
                return self.conv(x)

        conv_transpose2d_module = ConvTranspose2dModule()
        sample_inputs = (torch.randn(size=(1, 6, 40, 50), dtype=torch.float32),)

        self.lower_module_and_test_output(
            conv_transpose2d_module,
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_conv2d_dw(self):
        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=8,
                    out_channels=8,
                    kernel_size=3,
                    padding=1,
                    groups=8,
                    bias=True,
                )

            def forward(self, x):
                return self.conv(x)

        conv2d_module = Conv2dModule()
        sample_inputs = (torch.randn(size=(1, 8, 72, 96), dtype=torch.float32),)

        self.lower_module_and_test_output(
            conv2d_module,
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_conv2d_pw(self):
        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=8,
                    out_channels=8,
                    kernel_size=1,
                    padding=1,
                    groups=1,
                    bias=True,
                )

            def forward(self, x):
                return self.conv(x)

        conv2d_module = Conv2dModule()
        sample_inputs = (torch.randn(size=(1, 8, 72, 96), dtype=torch.float32),)

        self.lower_module_and_test_output(
            conv2d_module,
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )
