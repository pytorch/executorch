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
from torch.export import export, ExportedProgram

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
    ):
        """
        Helper testing function that takes a torch.nn.Module and lowers it to Vulkan with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """
        program: ExportedProgram = export(model, sample_inputs)
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
        model_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(add_module, model_inputs)

    def test_vulkan_backend_internal_data(self):
        class InternalDataModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.rand(size=(2, 3), dtype=torch.float32)

            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                z = z + self.weight
                return z

        internal_data_module = InternalDataModule()
        model_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(internal_data_module, model_inputs)

    def test_vulkan_backend_sub(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x - y
                z = z - x
                z = z - x
                return z

        sub_module = SubModule()
        model_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(sub_module, model_inputs)

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
        model_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(mul_module, model_inputs)

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
        model_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(div_module, model_inputs)

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
        model_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(arithmetic_module, model_inputs)

    def test_vulkan_backend_pow(self):
        class PowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.pow(x, y)
                return z

        pow_module = PowModule()
        model_inputs = (
            torch.rand(size=(2, 3), dtype=torch.float32),
            torch.rand(size=(2, 3), dtype=torch.float32),
        )

        self.lower_module_and_test_output(pow_module, model_inputs)

    def test_vulkan_backend_partial(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.offset_1 = self.weight = torch.rand(
                    size=(2, 10), dtype=torch.float32
                )
                self.offset_2 = self.weight = torch.rand(
                    size=(2, 10), dtype=torch.float32
                )

            def forward(self, x):
                return self.linear(x + self.offset_1) - self.offset_2

        model = SimpleModel()
        model_inputs = (torch.rand(size=(2, 10), dtype=torch.float32),)

        self.lower_module_and_test_output(model, model_inputs)
