# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import unittest
from typing import Tuple

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
    def assert_outputs_equal(self, model_output, ref_output, atol=1e-03, rtol=1e-03):
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
            for i in range(len(ref_output)):
                self.assertTrue(
                    torch.allclose(model_output[i], ref_output[i], atol=atol, rtol=rtol)
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
    ):
        """
        Helper testing function that takes a torch.nn.Module and lowers it to Vulkan with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """
        program: ExportedProgram = export(
            model, sample_inputs, dynamic_shapes=dynamic_shapes
        )
        edge_program: EdgeProgramManager = to_edge(program)
        edge_program = edge_program.to_backend(VulkanPartitioner())

        executorch_program = edge_program.to_executorch()

        self.assertEqual(
            executorch_program.executorch_program.execution_plan[0].delegates[0].id,
            VulkanBackend.__name__,
        )

        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        inputs_flattened, _ = tree_flatten(sample_inputs)

        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = model(*sample_inputs)

        self.assert_outputs_equal(model_output, ref_output, atol=atol, rtol=rtol)

        if test_inputs is not None:
            for test_input in test_inputs:
                # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
                test_inputs_flattened, _ = tree_flatten(test_input)
                model_output = executorch_module.run_method(
                    "forward", tuple(test_inputs_flattened)
                )
                ref_output = model(*test_input)

                self.assert_outputs_equal(
                    model_output, ref_output, atol=atol, rtol=rtol
                )

    def test_vulkan_backend_add(self):
        # This test is the simplest test by manually lowering some submodules, we can use paritioner for auto detecting lowerable parts
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                return z

        add_module = AddModule()
        sample_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
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
