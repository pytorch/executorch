# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import ctypes
import unittest
from typing import Tuple

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema

import torch

from executorch.backends.transforms.convert_dtype_pass import I64toI32

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend

from executorch.exir import EdgeCompileConfig
from torch.export import Dim, export, ExportedProgram

ctypes.CDLL("libvulkan.so.1")


from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten


class TestBackends(unittest.TestCase):
    _edge_compile_config: EdgeCompileConfig = EdgeCompileConfig(
        _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
    )

    def assert_outputs_equal(
        self,
        model_output,
        ref_output,
        atol=1e-03,
        rtol=1e-03,
        first_output_only=False,
        equal_nan=True,
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
                    torch.allclose(
                        model_output[0],
                        ref_output[0],
                        atol=atol,
                        rtol=rtol,
                        equal_nan=equal_nan,
                    )
                )
            else:
                for i in range(len(ref_output)):
                    self.assertTrue(
                        torch.allclose(
                            model_output[i],
                            ref_output[i],
                            atol=atol,
                            rtol=rtol,
                            equal_nan=equal_nan,
                        )
                    )
        else:
            # If one output, eager returns tensor while executor tuple of size 1
            self.assertTrue(
                torch.allclose(
                    model_output[0],
                    ref_output,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=equal_nan,
                )
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

            # At least model should run in eager mode.
            model.eval()
            model(*sample_inputs)

            program: ExportedProgram = export(
                model, sample_inputs, dynamic_shapes=dynamic_shapes
            )

            edge_program = to_edge_transform_and_lower(
                program,
                transform_passes=[
                    I64toI32(self._edge_compile_config._skip_dim_order),
                ],
                partitioner=[VulkanPartitioner(compile_options)],
            )
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

    def test_vulkan_backend_add_int(self):
        class AddIntModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                return z

        add_int_module = AddIntModule()
        sample_inputs = (
            torch.randint(low=-100, high=100, size=(2, 3), dtype=torch.int32),
            torch.randint(low=-100, high=100, size=(2, 3), dtype=torch.int32),
        )

        self.lower_module_and_test_output(add_int_module, sample_inputs)

    def test_vulkan_backend_zero_dim_tensor(self):
        class ZeroDimModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.zero = torch.full([], 1.3, dtype=torch.float32)

            def forward(self, x):
                return x + self.zero

        internal_data_module = ZeroDimModule()
        sample_inputs = (torch.rand(size=(2, 3), dtype=torch.float32),)
        self.lower_module_and_test_output(internal_data_module, sample_inputs)

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

    def lower_unary_module_and_test_output(self, module):
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

        self.lower_unary_module_and_test_output(ClampModule())

    def test_vulkan_backend_clamp_int(self):
        class ClampModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.clamp(x, min=-3)

        sample_inputs = (
            torch.randint(low=-100, high=100, size=(5, 5), dtype=torch.int32),
        )

        self.lower_module_and_test_output(ClampModule(), sample_inputs)

    def test_vulkan_backend_clamp_int64(self):
        class ClampModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.clamp(x, min=-3)

        sample_inputs = (
            torch.randint(low=-100, high=100, size=(5, 5), dtype=torch.int64),
        )

        self.lower_module_and_test_output(ClampModule(), sample_inputs)

    def test_vulkan_backend_cos(self):
        class CosModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.cos(x)

        self.lower_unary_module_and_test_output(CosModule())

    def test_vulkan_backend_hardtanh(self):
        class HardTanHModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tanh = torch.nn.Hardtanh(min_val=-3.14, max_val=6.28)

            def forward(self, x):
                return self.tanh(x)

        self.lower_unary_module_and_test_output(HardTanHModule())

    def test_vulkan_backend_exp(self):
        class ExpModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.exp(x)

        self.lower_unary_module_and_test_output(ExpModule())

    def test_vulkan_backend_neg(self):
        class NegModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.neg(x)

        self.lower_unary_module_and_test_output(NegModule())

    def test_vulkan_backend_sin(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        self.lower_unary_module_and_test_output(SinModule())

    def test_vulkan_backend_relu(self):
        class ReLUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.relu(x)

        self.lower_unary_module_and_test_output(ReLUModule())

    def test_vulkan_backend_sqrt(self):
        class SqrtModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sqrt(x)

        self.lower_unary_module_and_test_output(SqrtModule())

    def test_vulkan_backend_hardshrink(self):
        class HardshrinkModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hardshrink = torch.nn.Hardshrink(lambd=0.3)

            def forward(self, x):
                return self.hardshrink(x)

        self.lower_unary_module_and_test_output(HardshrinkModule())

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

    def test_vulkan_backend_avg_pool2d(self):
        class AvgPool2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool2d(
                    kernel_size=(4, 4),
                    stride=(4, 4),
                    padding=(0, 0),
                    ceil_mode=True,
                    count_include_pad=True,
                    divisor_override=None,
                )

            def forward(self, x):
                return self.avg_pool(x)

        avg_pool2d_module = AvgPool2dModule()
        sample_inputs = (torch.randn(5, 13, 55, 68),)

        batch = Dim("batch", max=8)
        dynamic_shapes = {"x": {0: batch}}
        test_inputs = [
            (torch.randn(3, 14, 15, 9),),
            (torch.randn(1, 1, 4, 6),),
            (torch.randn(5, 10, 50, 40),),
        ]
        self.lower_module_and_test_output(
            avg_pool2d_module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_abs(self):
        class AbsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.abs(x)

        self.lower_unary_module_and_test_output(AbsModule())

    def test_vulkan_backend_sigmoid(self):
        class SigmoidModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sigmoid(x)

        self.lower_unary_module_and_test_output(SigmoidModule())

    def test_vulkan_backend_tanh(self):
        class TanhModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.tanh(x)

        self.lower_unary_module_and_test_output(TanhModule())

    def test_vulkan_backend_linear(self):
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 64, bias=False)

            def forward(self, x):
                return self.linear(x)

        module = LinearModule()
        sample_inputs = (torch.rand(size=(32, 128), dtype=torch.float32),)
        batch = Dim("batch", max=32)
        dynamic_shapes = {"x": {0: batch}}

        test_inputs = [
            (torch.rand(15, 128),),
            (torch.rand(6, 128),),
            (torch.rand(30, 128),),
            (torch.rand(20, 128),),
            (torch.rand(19, 128),),
        ]

        self.lower_module_and_test_output(
            module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
        )

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

    def test_vulkan_backend_bmm(self):
        class BMMModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(size=(4, 4, 5), dtype=torch.float32)

            def forward(self, x):
                return torch.bmm(x, self.weight)

        module = BMMModule()
        sample_inputs = (torch.randn(size=(4, 3, 4), dtype=torch.float32),)

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

    def test_vulkan_backend_sum(self):
        class SumModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.sum(x, (), keepdim=True)
                x = torch.sum(x)
                return x

        module = SumModule()
        sample_inputs = (torch.rand(size=(3, 2, 7, 5), dtype=torch.float32),)

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

    def test_vulkan_backend_conv2d_bias_false(self):
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
                    bias=False,
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

    def test_vulkan_backend_conv1d(self):
        class Conv1dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    in_channels=20,
                    out_channels=10,
                    kernel_size=6,
                    stride=5,
                    padding=5,
                    dilation=3,
                    groups=5,
                    bias=True,
                )

            def forward(self, x):
                return self.conv(x)

        conv1d_module = Conv1dModule()
        sample_inputs = (torch.randn(size=(3, 20, 30), dtype=torch.float32),)

        self.lower_module_and_test_output(
            conv1d_module,
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_conv1d_bias_false(self):
        class Conv1dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    in_channels=6,
                    out_channels=6,
                    kernel_size=3,
                    groups=6,
                    bias=False,
                )

            def forward(self, x):
                return self.conv(x)

        conv1d_module = Conv1dModule()
        sample_inputs = (torch.randn(size=(1, 6, 7), dtype=torch.float32),)

        self.lower_module_and_test_output(
            conv1d_module,
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_native_layer_norm(self):
        class NativeLayerNormModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.native_layer_norm(
                    x, [5], torch.ones(5), torch.zeros(5), 1e-5
                )

        sample_inputs = (torch.randn(size=(3, 4, 5), dtype=torch.float32),)

        self.lower_module_and_test_output(
            NativeLayerNormModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_batch_norm(self):
        class BatchNormModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(num_features=3)

            def forward(self, x):
                return self.bn(x)

        sample_inputs = (torch.randn(size=(4, 3, 2, 5), dtype=torch.float32),)

        self.lower_module_and_test_output(
            BatchNormModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_full(self):
        class FullModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.full(x.shape, 42.0)

        class ZerosModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.zeros(x.shape)

        class OnesModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ones(x.shape)

        sample_inputs = (torch.randn(size=(2, 3, 4, 5), dtype=torch.float32),)

        self.lower_module_and_test_output(
            FullModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            ZerosModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            OnesModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_full_like(self):
        class FullLikeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.full_like(x, 42.0)

        class ZerosLikeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.zeros_like(x)

        class OnesLikeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ones_like(x)

        sample_inputs = (torch.randn(size=(2, 3, 4, 5), dtype=torch.float32),)

        self.lower_module_and_test_output(
            FullLikeModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            ZerosLikeModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            OnesLikeModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_upsample_nearest2d(self):
        class UpsampleNearest2d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

            def forward(self, x):
                return self.upsample(x)

        sample_inputs = (torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2),)

        self.lower_module_and_test_output(
            UpsampleNearest2d(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_minimum(self):
        class MinimumModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.minimum(x, y)

        sample_inputs = (
            torch.rand(size=(3, 5, 6, 4), dtype=torch.float32),
            torch.rand(size=(6, 4), dtype=torch.float32),
        )

        self.lower_module_and_test_output(
            MinimumModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_reshape(self):
        class ReshapeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.reshape(x, [-1, x.size(-1)])

        sample_inputs = (torch.randn(size=(5, 3, 4), dtype=torch.float32),)

        self.lower_module_and_test_output(
            ReshapeModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_view(self):
        class ViewModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.view([-1, x.size(-1)])

        sample_inputs = (torch.randn(size=(3, 2, 3, 4), dtype=torch.float32),)

        self.lower_module_and_test_output(
            ViewModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_view_int(self):
        class ViewModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.view([-1, x.size(-1)])

        sample_inputs = (torch.randint(size=(3, 6, 2, 7), high=100, dtype=torch.int32),)

        self.lower_module_and_test_output(
            ViewModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_unsqueeze(self):
        class UnsqueezeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.unsqueeze(x, 1)
                x = torch.unsqueeze(x, 0)
                return x

        sample_inputs = (torch.randn(size=(3,), dtype=torch.float32),)

        self.lower_module_and_test_output(
            UnsqueezeModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_squeeze(self):
        class SqueezeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.squeeze(x, 0)

        sample_inputs = (torch.randn(size=(1, 2, 2, 1), dtype=torch.float32),)

        self.lower_module_and_test_output(
            SqueezeModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_select(self):
        class SelectModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[0][3]

        sample_inputs = (torch.randn(size=(3, 6, 2, 7), dtype=torch.float32),)

        self.lower_module_and_test_output(
            SelectModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_permute_copy(self):
        class PermuteModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.permute(x, [3, 0, 2, 1])

        sample_inputs = (torch.randn(size=(3, 6, 2, 7), dtype=torch.float32),)

        self.lower_module_and_test_output(
            PermuteModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_permute_copy_int(self):
        class PermuteModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.permute(x, [3, 0, 2, 1])

        sample_inputs = (torch.randint(size=(3, 6, 2, 7), high=100, dtype=torch.int32),)

        self.lower_module_and_test_output(
            PermuteModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_cat(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z, w):
                return torch.cat([x, y, z, w], dim=1)

        sample_inputs = (
            torch.randn(size=(3, 6, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 1, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 9, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 3, 2, 7), dtype=torch.float32),
        )

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_cat_with_zero_size(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z, w):
                return torch.cat([x, y, z, w], dim=1)

        sample_inputs = (
            torch.randn(size=(3, 6, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 0, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 0, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 3, 2, 7), dtype=torch.float32),
        )

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_slice(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, 2:9:2, :]

        sample_inputs = (torch.randn(size=(3, 13, 7, 3), dtype=torch.float32),)

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_split_with_sizes(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, (3, 6, 1, 3), dim=1)

        sample_inputs = (torch.randn(size=(3, 13, 7, 3), dtype=torch.float32),)

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_split_tensor(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.tensor_split(x, 2, dim=1)

        sample_inputs = (torch.randn(size=(3, 14, 7, 3), dtype=torch.float32),)

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_clone(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.clone(x)

        sample_inputs = (torch.randn(size=(3, 14, 7, 3), dtype=torch.float32),)

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_constant_pad_nd(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.pad(x, (1, 2, 3, 4, 5, 6), "constant", 24.2)

        sample_inputs = (torch.randn(size=(3, 7, 5, 11), dtype=torch.float32),)

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_repeat(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.repeat([2, 3, 1, 2])

        sample_inputs = (torch.randn(size=(3, 7, 5, 9), dtype=torch.float32),)

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_t_default(self):
        # aten.permute_copy.default is not enabled yet in partitioner
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # torch.t is actually exported as aten::permute.
                return torch.t(x)

        sample_inputs = (torch.randn(size=(3, 14), dtype=torch.float32),)

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_softmax(self):
        class SoftmaxModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x.softmax(dim=0)
                x = x.softmax(dim=1)
                x = x.softmax(dim=2)
                return x

        sample_inputs = (torch.randn(size=(3, 2, 7), dtype=torch.float32),)

        self.lower_module_and_test_output(
            SoftmaxModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_logsoftmax(self):
        class LogSoftmaxModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x.log_softmax(dim=0)
                x = x.log_softmax(dim=1)
                x = x.log_softmax(dim=2)
                return x

        sample_inputs = (torch.randn(size=(3, 2, 7), dtype=torch.float32),)

        self.lower_module_and_test_output(
            LogSoftmaxModule(),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_gelu(self):
        class GeluModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = torch.nn.GELU(approximate="tanh")

            def forward(self, x):
                return self.gelu(x)

        self.lower_unary_module_and_test_output(GeluModule())

    def test_vulkan_backend_mean(self):
        class MeanModule(torch.nn.Module):
            def __init__(self, dims, keepdim=True):
                super().__init__()
                self.dims = dims
                self.keepdim = keepdim

            def forward(self, x):
                return torch.mean(x, self.dims, keepdim=self.keepdim)

        sample_inputs = (
            torch.arange(end=2 * 3 * 2 * 5, dtype=torch.float32).reshape(2, 3, 2, 5),
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[-1, -2]),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[1]),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[0, 1, 2, 3]),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[-1, -2], keepdim=False),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[1], keepdim=False),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_index_select_int(self):
        class IndexSelectModule(torch.nn.Module):
            def __init__(self, dim, indices):
                super().__init__()
                self.dim = dim
                self.index = torch.tensor(indices)

            def forward(self, x):
                return torch.index_select(x, self.dim, self.index)

        sample_inputs = (torch.arange(96).reshape(2, 8, 2, 3),)

        self.lower_module_and_test_output(
            IndexSelectModule(dim=1, indices=[2, 3, 5, 6, 7]),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_index_select(self):
        class IndexSelectModule(torch.nn.Module):
            def __init__(self, dim, indices):
                super().__init__()
                self.dim = dim
                self.index = torch.tensor(indices)

            def forward(self, x):
                return torch.index_select(x, self.dim, self.index)

        sample_inputs = (torch.arange(144).reshape(12, 1, 3, 4).float(),)

        self.lower_module_and_test_output(
            IndexSelectModule(dim=0, indices=[1, 3, 5, 7, 8, 9, 10, 11, 2, 3]),
            sample_inputs,
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_arange_int(self):
        class ArangeModule(torch.nn.Module):
            def __init__(self, input):
                super().__init__()
                self.input = input

            def forward(self, x):
                return torch.arange(*self.input, dtype=torch.int32)

        # `torch.arange` could take one, two or three arguments as input.
        # If only one argument is provided, it will be interpreted as `end`.
        # If two arguments are provided, the first one will be interpreted as `start`
        # and the second one will be interpreted as `end`.
        # If three arguments are provided, the first one will be interpreted as `start`,
        # the second one will be interpreted as `end` and the third one will be
        # interpreted as `step`.
        inputs = [
            [1],
            [-3, 5],
            [1, 11, 2],
            [12, 1, -2],
        ]
        for i in inputs:
            self.lower_module_and_test_output(
                ArangeModule(i),
                (torch.randn(size=(1,), dtype=torch.float32),),  # dummy input
                memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
            )

    def test_vulkan_backend_arange_float(self):
        class ArangeModule(torch.nn.Module):
            def __init__(self, input):
                super().__init__()
                self.input = input

            def forward(self, x):
                return torch.arange(*self.input)

        inputs = [
            [1.5],
            [-3, 5.0],
            [1.0, 11, 2],
            [12, 1, -2.0],
        ]
        for i in inputs:
            self.lower_module_and_test_output(
                ArangeModule(i),
                (torch.randn(size=(1,), dtype=torch.float32),),  # dummy input
                memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
            )

    def test_vulkan_backend_arange_int64(self):
        class ArangeModule(torch.nn.Module):
            def __init__(self, input):
                super().__init__()
                self.input = input

            def forward(self, x):
                return torch.arange(*self.input)

        inputs = [
            [1],
            [-3, 5],
            [1, 11, 2],
            [12, 1, -2],
            [1.5],
            [-3, 5.0],
            [1.0, 11, 2],
            [12, 1, -2.0],
        ]
        for i in inputs:
            self.lower_module_and_test_output(
                ArangeModule(i),
                (torch.randn(size=(1,), dtype=torch.float32),),  # dummy input
                memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
            )
            self.lower_module_and_test_output(
                ArangeModule(i),
                (torch.randint(low=-100, high=100, size=(5, 5)),),  # dummy input
                memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
            )

    def test_vulkan_backend_embedding_1d(self):
        class EmbeddingModule(torch.nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.embedding = embedding

            def forward(self, x):
                return self.embedding(x)

        self.lower_module_and_test_output(
            EmbeddingModule(torch.nn.Embedding(5, 4)),
            (torch.tensor([0, 1, 0, 4, 2, 0]),),
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_embedding_2d(self):
        class EmbeddingModule(torch.nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.embedding = embedding

            def forward(self, x):
                return self.embedding(x)

        self.lower_module_and_test_output(
            EmbeddingModule(torch.nn.Embedding(5, 4)),
            (torch.tensor([[0, 1, 0], [4, 2, 0]]),),
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_embedding_3d(self):
        class EmbeddingModule(torch.nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.embedding = embedding

            def forward(self, x):
                return self.embedding(x)

        self.lower_module_and_test_output(
            EmbeddingModule(torch.nn.Embedding(5, 4)),
            (torch.tensor([[[0, 1], [0, 1]], [[4, 2], [3, 3]]]),),
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_conv_with_clamp(self):
        class ConvWithClampModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(6, 8, 3, 3)
                self.bias = torch.randn(8)
                self.stride = (1, 2)
                self.padding = (2, 3)
                self.dilation = (1, 1)
                self.transposed = True
                self.output_padding = (0, 1)
                self.groups = 1
                self.output_min = 0
                self.output_max = 10

            def forward(self, x):
                return torch.ops.et_vk.conv_with_clamp(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.transposed,
                    self.output_padding,
                    self.groups,
                    self.output_min,
                    self.output_max,
                )

        self.lower_module_and_test_output(
            ConvWithClampModule(),
            (torch.randn(size=(1, 6, 40, 50), dtype=torch.float32),),
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )

    def test_vulkan_backend_grid_priors(self):
        class GridPriorsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.et_vk.grid_priors(
                    x,
                    stride=8,
                    offset=0.5,
                )

        self.lower_module_and_test_output(
            GridPriorsModule(),
            (torch.rand(size=[1, 5, 2, 3]),),
            memory_layouts=[vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED],
        )
