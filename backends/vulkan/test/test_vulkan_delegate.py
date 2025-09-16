# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import ctypes
import unittest
from typing import Tuple

import executorch.backends.vulkan.test.utils as test_utils

import torch

from executorch.backends.transforms.convert_dtype_pass import I64toI32

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.backends.vulkan.vulkan_preprocess import VulkanBackend

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten
from torch.export import Dim, export, ExportedProgram

from torchao.quantization.granularity import PerGroup

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
from torchao.utils import unwrap_tensor_subclass

try:
    ctypes.CDLL("libvulkan.so.1")
except:
    pass


def lower_module(
    model: torch.nn.Module, sample_inputs: Tuple[torch.Tensor], dynamic_shapes=None
) -> EdgeProgramManager:
    compile_options = {}
    if dynamic_shapes is not None:
        compile_options["require_dynamic_shapes"] = True

    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # TODO(T182928844): Delegate dim order op to backend.
    )

    program: ExportedProgram = export(
        model, sample_inputs, dynamic_shapes=dynamic_shapes, strict=True
    )

    edge_program = to_edge_transform_and_lower(
        program,
        compile_config=edge_compile_config,
        transform_passes=[
            I64toI32(edge_compile_config._skip_dim_order),
        ],
        partitioner=[VulkanPartitioner(compile_options)],
    )

    return edge_program


def quantize_and_lower_module(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
    quantizer: Quantizer,
    dynamic_shapes=None,
) -> EdgeProgramManager:
    compile_options = {}
    if dynamic_shapes is not None:
        compile_options["require_dynamic_shapes"] = True

    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # TODO(T182928844): Delegate dim order op to backend.
    )

    program = export(
        model, sample_inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()

    program = prepare_pt2e(program, quantizer)
    # Calibrate
    program(*sample_inputs)

    program = convert_pt2e(program)

    program = export(program, sample_inputs, dynamic_shapes=dynamic_shapes)

    edge_program = to_edge_transform_and_lower(
        program,
        compile_config=edge_compile_config,
        transform_passes=[
            I64toI32(edge_compile_config._skip_dim_order),
        ],
        partitioner=[VulkanPartitioner(compile_options)],
    )

    return edge_program


class TestVulkanBackend(unittest.TestCase):
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
                result = torch.allclose(
                    model_output[0],
                    ref_output[0],
                    atol=atol,
                    rtol=rtol,
                    equal_nan=equal_nan,
                )
                if not result:
                    test_utils.print_tensor_comparison_errors(
                        model_output[0], ref_output[0], atol, rtol
                    )
                self.assertTrue(result)
            else:
                for i in range(len(ref_output)):
                    result = torch.allclose(
                        model_output[i],
                        ref_output[i],
                        atol=atol,
                        rtol=rtol,
                        equal_nan=equal_nan,
                    )
                    if not result:
                        print(f"\n=== Output {i} comparison failed ===")
                        test_utils.print_tensor_comparison_errors(
                            model_output[i], ref_output[i], atol, rtol
                        )
                    self.assertTrue(result)
        else:
            # If one output, eager returns tensor while executor tuple of size 1
            result = torch.allclose(
                model_output[0],
                ref_output,
                atol=atol,
                rtol=rtol,
                equal_nan=equal_nan,
            )
            if not result:
                test_utils.print_tensor_comparison_errors(
                    model_output[0], ref_output, atol, rtol
                )
            self.assertTrue(result)

    def check_no_delegation(self, et_program: ExecutorchProgramManager):
        self.assertEqual(
            len(et_program.executorch_program.execution_plan[0].delegates),
            0,
        )
        return

    def check_vk_delegation(self, et_program: ExecutorchProgramManager):
        self.assertEqual(
            et_program.executorch_program.execution_plan[0].delegates[0].id,
            VulkanBackend.__name__,
        )

    def run_delegated_model_and_check_output(
        self,
        et_program: ExecutorchProgramManager,
        model: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        atol=1e-03,
        rtol=1e-01,
        test_inputs=None,
        first_output_only=False,
    ):
        executorch_module = _load_for_executorch_from_buffer(et_program.buffer)
        inputs_flattened, _ = tree_flatten(sample_inputs)

        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
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

    def lower_module_and_test_output(
        self,
        model: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        atol=1e-03,
        rtol=1e-01,
        dynamic_shapes=None,
        test_inputs=None,
        first_output_only=False,
        expect_no_delegates=False,
    ):
        """
        Helper testing function that takes a torch.nn.Module and lowers it to Vulkan with
        the given sample inputs. It then runs the lowered module and compares its
        outputs with the outputs of the eager module.
        """

        # Validate that the model can execute in eager mode
        model.eval()
        model(*sample_inputs)

        edge_program = lower_module(model, sample_inputs, dynamic_shapes=dynamic_shapes)

        et_program = edge_program.to_executorch()

        if expect_no_delegates:
            self.check_no_delegation(et_program)
            return

        self.check_vk_delegation(et_program)

        self.run_delegated_model_and_check_output(
            et_program,
            model,
            sample_inputs,
            atol,
            rtol,
            test_inputs=test_inputs,
            first_output_only=first_output_only,
        )

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

        sample_inputs = (
            torch.rand(size=(4, 5, 2, 3), dtype=torch.float32),
            torch.rand(size=(4, 5, 2, 3), dtype=torch.float32),
            torch.rand(
                size=(2, 3), dtype=torch.float32
            ),  # test broadcasting on packed dim
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
                inter1 = torch.add(x, y, alpha=2)
                inter2 = torch.add(x, y, alpha=3.14)
                inter3 = inter1 * self.weight
                inter4 = inter2 * self.weight
                return inter4 - inter3

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

    @unittest.skip(
        "Currently this test is failing due to weird partitioning because the eq scalar"
        "operator is not supported yet. Re-enable when the operator is supported."
    )
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

    @unittest.skip(
        "Reduce shader does not support multiple reduction axes at the moment"
    )
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
        )

    @unittest.skip(
        "Reduce shader does not support multiple reduction axes at the moment"
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
        )

    @unittest.skip("layer norm compute shader not working with swiftshader")
    def test_vulkan_backend_native_layer_norm(self):
        class NativeLayerNormModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm(5)

            def forward(self, x):
                return self.layer_norm(x)

        sample_inputs = (torch.randn(size=(3, 4, 5), dtype=torch.float32),)

        self.lower_module_and_test_output(
            NativeLayerNormModule(),
            sample_inputs,
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
        )

        self.lower_module_and_test_output(
            ZerosModule(),
            sample_inputs,
        )

        self.lower_module_and_test_output(
            OnesModule(),
            sample_inputs,
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
        )

        self.lower_module_and_test_output(
            ZerosLikeModule(),
            sample_inputs,
        )

        self.lower_module_and_test_output(
            OnesLikeModule(),
            sample_inputs,
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
        )

    def test_vulkan_backend_cat(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                return torch.cat([x, y, z], dim=1)

        sample_inputs = (
            torch.randn(size=(3, 6, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 1, 2, 7), dtype=torch.float32),
            torch.randn(size=(3, 9, 2, 7), dtype=torch.float32),
        )

        self.lower_module_and_test_output(
            TestModule(),
            sample_inputs,
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
        )

    @unittest.skip(
        "Softmax shader with shared memory does not work with swiftshader due to potential swiftshader bug"
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
        )

    @unittest.skip(
        "Softmax shader with shared memory does not work with swiftshader due to potential swiftshader bug"
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
        )

    def test_vulkan_backend_gelu(self):
        class GeluModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = torch.nn.GELU(approximate="tanh")

            def forward(self, x):
                return self.gelu(x)

        self.lower_unary_module_and_test_output(GeluModule())

    @unittest.skip(
        "Reduce shader does not support multiple reduction axes at the moment"
    )
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
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[1]),
            sample_inputs,
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[0, 1, 2, 3]),
            sample_inputs,
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[-1, -2], keepdim=False),
            sample_inputs,
        )

        self.lower_module_and_test_output(
            MeanModule(dims=[1], keepdim=False),
            sample_inputs,
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
            )
            self.lower_module_and_test_output(
                ArangeModule(i),
                (torch.randint(low=-100, high=100, size=(5, 5)),),  # dummy input
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
        )

    # def test_vulkan_backend_conv_with_dim_order(self):
    #     class Conv2dSequential(torch.nn.Module):
    #         def __init__(self, bias=True, channel_last=False):
    #             super().__init__()
    #             self.first = torch.nn.Conv2d(
    #                 in_channels=1,
    #                 out_channels=3,
    #                 kernel_size=(3, 3),
    #                 padding=1,
    #                 bias=bias,
    #             )
    #             self.second = torch.nn.Conv2d(
    #                 in_channels=3,
    #                 out_channels=2,
    #                 kernel_size=(3, 3),
    #                 padding=1,
    #                 bias=bias,
    #             )

    #         def forward(self, x):
    #             x = x.to(memory_format=torch.channels_last)
    #             return self.second(self.first(x))

    #     self.lower_module_and_test_output(
    #         Conv2dSequential(),
    #         (torch.rand(size=[1, 1, 3, 3]),),
    #
    #     )

    def test_vulkan_backend_flip(self):
        class FlipModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.flip(x, [0, 1, 2, 3])

        self.lower_module_and_test_output(
            FlipModule(),
            (torch.arange(48).reshape(2, 3, 4, 2),),
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
        )

    def test_vulkan_backend_large_linear_layer(self):
        class LinearModel(torch.nn.Module):
            def __init__(self, large_out_channels: int) -> None:
                super(LinearModel, self).__init__()
                self.fc0 = torch.nn.Linear(1024, 128)
                self.fc1 = torch.nn.Linear(128, large_out_channels)

            def forward(self, x: torch.Tensor):
                x = self.fc0(x)
                out = self.fc1(x)
                return out

        large_out_channels = 2**16

        self.lower_module_and_test_output(
            LinearModel(large_out_channels),
            (torch.ones(1024),),
        )

    def test_vulkan_backend_sym_size_int(self):
        """
        Test the sym_size.int operator with a model that:
        1. Takes an input tensor with shape [1, M, K]
        2. Reshapes it to [M, K]
        3. Applies a linear layer
        4. Reshapes the output back to [1, M, N]
        """
        K = 64  # Input feature dimension
        N = 32  # Output feature dimension

        class SymSizeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(K, N)

            def forward(self, x):
                M = x.size(1)

                reshaped = torch.reshape(x, [M, K])
                output = self.linear(reshaped)
                return torch.reshape(output, [1, M, N])

        sample_inputs = (torch.randn(1, 64, K),)

        batch = Dim("batch", min=1, max=128)
        dynamic_shapes = {"x": {1: batch}}

        test_inputs = [
            (torch.randn(1, 32, K),),
            (torch.randn(1, 96, K),),
            (torch.randn(1, 128, K),),
        ]

        self.lower_module_and_test_output(
            SymSizeModel(),
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
        )

    def test_select_last_height_dynamic_shapes(self):
        """
        Test selecting the last element along the height dimension with dynamic shapes.
        The height dimension (dim=1) is variable.
        """

        class SelectLastHeightModule(torch.nn.Module):
            """
            Module that selects the last element along the height dimension (dim=1) of a 3D tensor.
            This is equivalent to the operation: x[:, -1, :]
            """

            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Select the last element along dimension 1 (height)
                return x[:, -1, :]

        # Create the module
        module = SelectLastHeightModule()

        # Create sample inputs with a specific shape
        # Shape: [batch_size, height, width]
        sample_inputs = (torch.arange(1, 61).reshape(2, 10, 3).float(),)

        # Define dynamic shapes for the height dimension
        height = Dim("height", min=1, max=10)
        dynamic_shapes = {"x": {1: height}}

        # Create test inputs with different heights
        test_inputs = [
            (torch.arange(1, 7).reshape(2, 1, 3).float(),),  # Minimum height
            (torch.arange(1, 19).reshape(2, 3, 3).float(),),  # Small height
            (torch.arange(1, 43).reshape(2, 7, 3).float(),),  # Medium height
            (torch.arange(1, 31).reshape(2, 5, 3).float(),),  # Maximum height
        ]

        # Use the testing infrastructure from TestVulkanBackend
        test_backend = TestVulkanBackend()
        test_backend.lower_module_and_test_output(
            module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
        )

    def test_vulkan_backend_group_norm(self):
        class ConvGroupNormModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Conv2d: 3 input channels -> 16 output channels
                self.conv = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                # GroupNorm: 4 groups for 16 channels (16 % 4 == 0)
                self.group_norm = torch.nn.GroupNorm(
                    num_groups=4,
                    num_channels=16,
                    eps=1e-5,
                    affine=True,
                )

            def forward(self, x):
                x = self.conv(x)
                x = self.group_norm(x)
                return x

        # Create sample inputs: [batch, channels, height, width]
        sample_inputs = (torch.randn(size=(1, 3, 32, 32), dtype=torch.float32),)

        # Test with static shapes first
        self.lower_module_and_test_output(
            ConvGroupNormModule(),
            sample_inputs,
        )

    def test_vulkan_backend_group_norm_different_groups(self):
        class GroupNormModule(torch.nn.Module):
            def __init__(self, num_groups, num_channels):
                super().__init__()
                self.group_norm = torch.nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=num_channels,
                    eps=1e-5,
                    affine=True,
                )

            def forward(self, x):
                return self.group_norm(x)

        # Test different group configurations
        test_configs = [
            (2, 8),  # 2 groups, 8 channels
            (4, 16),  # 4 groups, 16 channels
            (8, 32),  # 8 groups, 32 channels
        ]

        for num_groups, num_channels in test_configs:
            with self.subTest(num_groups=num_groups, num_channels=num_channels):
                sample_inputs = (
                    torch.randn(size=(2, num_channels, 16, 16), dtype=torch.float32),
                )

                self.lower_module_and_test_output(
                    GroupNormModule(num_groups, num_channels),
                    sample_inputs,
                )

    def test_vulkan_backend_full_quantization_workflow(self):
        class FullQuantizationWorkflowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Step 1: Choose quantization parameters per tensor
                scale, zero_point = (
                    torch.ops.quantized_decomposed.choose_qparams.tensor(
                        x,
                        quant_min=-2147483648,  # int32 min
                        quant_max=2147483647,  # int32 max
                        eps=1e-5,
                        dtype=torch.int32,
                    )
                )

                # Step 2: Quantize using the calculated parameters
                quantized = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
                    x,
                    scale,
                    zero_point,
                    quant_min=-2147483648,  # int32 min
                    quant_max=2147483647,  # int32 max
                    dtype=torch.int32,
                )

                # Step 3: Dequantize back to float
                dequantized = (
                    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
                        quantized,
                        scale,
                        zero_point,
                        quant_min=-2147483648,  # int32 min
                        quant_max=2147483647,  # int32 max
                        dtype=torch.int32,
                    )
                )

                return dequantized

        full_workflow_module = FullQuantizationWorkflowModule()
        sample_inputs = (torch.rand(size=(2, 3, 4), dtype=torch.float32),)

        # Use higher tolerance since quantization introduces some error
        self.lower_module_and_test_output(
            full_workflow_module, sample_inputs, atol=5e-3, rtol=5e-3
        )

    def test_vulkan_backend_full_per_token_quantization_workflow(self):
        class FullPerTokenQuantizationWorkflowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Step 1: Choose quantization parameters per token
                scale, zero_point = (
                    torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                        x,
                        dtype=torch.int32,
                    )
                )

                # Step 2: Quantize using the calculated parameters per token
                quantized = torch.ops.quantized_decomposed.quantize_per_token.default(
                    x,
                    scale,
                    zero_point,
                    quant_min=-2147483648,  # int32 min
                    quant_max=2147483647,  # int32 max
                    dtype=torch.int32,
                )

                # Step 3: Dequantize back to float per token
                dequantized = (
                    torch.ops.quantized_decomposed.dequantize_per_token.default(
                        quantized,
                        scale,
                        zero_point,
                        quant_min=-2147483648,  # int32 min
                        quant_max=2147483647,  # int32 max
                        dtype=torch.int32,
                        output_dtype=torch.float32,
                    )
                )

                return dequantized

        full_per_token_workflow_module = FullPerTokenQuantizationWorkflowModule()
        sample_inputs = (torch.rand(size=(6, 4), dtype=torch.float32),)

        # Use higher tolerance since quantization introduces some error
        self.lower_module_and_test_output(
            full_per_token_workflow_module, sample_inputs, atol=5e-3, rtol=5e-3
        )

    def test_vulkan_backend_different_required_reprs(self):
        class ComplexModule(torch.nn.Module):
            """
            This Module tests the tag memory metadata pass. The first few ops executed
            are binary ops, which don't require any specific representation for input
            and output tensors.

            This is followed by a linear layer, which requires the input tensor to be
            width packed.

            Three linear layer outputs are then concatenated, and the result is passed
            to a convolution layer which requires channels packing. Finally, group norm
            is called and the output is postprocessed by a binary op before returning.

            In addition to requiring memory layout transitions between the linear and
            conv stages, the module also contains ops which have "non-standard"
            torch.fx.Nodes; cat will contain an argument node that is a list of nodes,
            and group norm's node will be associated with multiple output tensors.
            """

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.conv = torch.nn.Conv2d(
                    in_channels=3,  # Assuming concatenation triples the channels
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                )
                self.group_norm = torch.nn.GroupNorm(num_groups=4, num_channels=16)

            def forward(self, x, a, b, c, d):
                w = a + b
                y = a + c
                z = a + d

                b1 = x + y
                b2 = x + z
                b3 = x + w

                l1 = self.linear(b1).unsqueeze(0)
                l2 = self.linear(b2).unsqueeze(0)
                l3 = self.linear(b3).unsqueeze(0)

                concat = torch.cat([l1, l2, l3], dim=0)  # Concatenate along channels
                conv = self.conv(concat + a)
                g = self.group_norm(conv.unsqueeze(0))
                return g + x

        complex_module = ComplexModule()
        sample_inputs = (
            torch.rand(size=(10, 10), dtype=torch.float32),  # x
            torch.rand(size=(10, 10), dtype=torch.float32),  # a
            torch.rand(size=(10, 10), dtype=torch.float32),  # b
            torch.rand(size=(10, 10), dtype=torch.float32),  # c
            torch.rand(size=(10, 10), dtype=torch.float32),  # d
        )

        self.lower_module_and_test_output(complex_module, sample_inputs)

    def test_vulkan_backend_cat_different_reprs(self):
        class CustomComplexModule(torch.nn.Module):
            """
            This test validates that the memory metadata tagging pass can handle
            transitioning arguments to the cat operator. Linear layers require width
            packing, while conv layers require channels packing. Before executing the
            cat operator, all input tensors should use the same representation.
            """

            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.conv = torch.nn.Conv2d(
                    in_channels=4,  # Assuming input b has 3 channels
                    out_channels=8,
                    kernel_size=3,
                    padding=1,
                )

            def forward(self, a, b):
                x1 = self.linear1(a).unsqueeze(0)
                x2 = self.linear2(a).unsqueeze(0)
                y = self.conv(b)
                return torch.cat([x1, x2, y], dim=0)

        custom_complex_module = CustomComplexModule()
        sample_inputs = (
            torch.rand(size=(10, 10), dtype=torch.float32),  # a
            torch.rand(size=(4, 10, 10), dtype=torch.float32),  # b
        )

        self.lower_module_and_test_output(custom_complex_module, sample_inputs)

    def test_vulkan_backend_cat_width_dynamic_shapes(self):
        class CatWidthModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, x3, x4, x5, x6):
                return torch.cat([x1, x2, x3, x4, x5, x6], dim=3)

        cat_width_module = CatWidthModule()

        # Create 6 tensors with different widths but same batch, channel, and height dimensions
        sample_inputs = (
            torch.randn(size=(2, 3, 4, 5), dtype=torch.float32),  # width=5
            torch.randn(size=(2, 3, 4, 3), dtype=torch.float32),  # width=3
            torch.randn(size=(2, 3, 4, 7), dtype=torch.float32),  # width=7
            torch.randn(size=(2, 3, 4, 2), dtype=torch.float32),  # width=2
            torch.randn(size=(2, 3, 4, 4), dtype=torch.float32),  # width=4
            torch.randn(size=(2, 3, 4, 6), dtype=torch.float32),  # width=6
        )

        # Define dynamic shapes for the width dimension (dim=3) for each input
        width1 = Dim("width1", min=1, max=10)
        width2 = Dim("width2", min=1, max=10)
        width3 = Dim("width3", min=1, max=10)
        width4 = Dim("width4", min=1, max=10)
        width5 = Dim("width5", min=1, max=10)
        width6 = Dim("width6", min=1, max=10)

        dynamic_shapes = {
            "x1": {3: width1},
            "x2": {3: width2},
            "x3": {3: width3},
            "x4": {3: width4},
            "x5": {3: width5},
            "x6": {3: width6},
        }

        # Create test inputs with different width combinations
        test_inputs = [
            (
                torch.randn(2, 3, 4, 2),  # width=2
                torch.randn(2, 3, 4, 1),  # width=1
                torch.randn(2, 3, 4, 3),  # width=3
                torch.randn(2, 3, 4, 1),  # width=1
                torch.randn(2, 3, 4, 2),  # width=2
                torch.randn(2, 3, 4, 4),  # width=4
            ),
            (
                torch.randn(2, 3, 4, 8),  # width=8
                torch.randn(2, 3, 4, 2),  # width=2
                torch.randn(2, 3, 4, 1),  # width=1
                torch.randn(2, 3, 4, 3),  # width=3
                torch.randn(2, 3, 4, 5),  # width=5
                torch.randn(2, 3, 4, 1),  # width=1
            ),
            (
                torch.randn(2, 3, 4, 1),  # width=1
                torch.randn(2, 3, 4, 9),  # width=9
                torch.randn(2, 3, 4, 2),  # width=2
                torch.randn(2, 3, 4, 4),  # width=4
                torch.randn(2, 3, 4, 1),  # width=1
                torch.randn(2, 3, 4, 3),  # width=3
            ),
        ]

        self.lower_module_and_test_output(
            cat_width_module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
        )

    def test_vulkan_backend_cat_channels_dynamic_shapes(self):
        class CatChannelsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, x3, x4, x5, x6):
                return torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

        cat_channels_module = CatChannelsModule()

        # Create 6 tensors with different channel counts but same batch, height, and width dimensions
        sample_inputs = (
            torch.randn(size=(2, 8, 8, 6), dtype=torch.float32),  # channels=4
            torch.randn(size=(2, 8, 8, 6), dtype=torch.float32),  # channels=2
            torch.randn(size=(2, 8, 8, 6), dtype=torch.float32),  # channels=6
            torch.randn(size=(2, 8, 8, 6), dtype=torch.float32),  # channels=1
            torch.randn(size=(2, 8, 8, 6), dtype=torch.float32),  # channels=3
            torch.randn(size=(2, 8, 8, 6), dtype=torch.float32),  # channels=5
        )

        # Define dynamic shapes for the channels dimension (dim=1) for each input
        channels1 = Dim("channels1", min=1, max=8)
        channels2 = Dim("channels2", min=1, max=8)
        channels3 = Dim("channels3", min=1, max=8)
        channels4 = Dim("channels4", min=1, max=8)
        channels5 = Dim("channels5", min=1, max=8)
        channels6 = Dim("channels6", min=1, max=8)

        dynamic_shapes = {
            "x1": {1: channels1},
            "x2": {1: channels2},
            "x3": {1: channels3},
            "x4": {1: channels4},
            "x5": {1: channels5},
            "x6": {1: channels6},
        }

        # Create test inputs with different channel combinations
        test_inputs = [
            (
                torch.randn(2, 1, 8, 6),  # channels=1
                torch.randn(2, 2, 8, 6),  # channels=2
                torch.randn(2, 1, 8, 6),  # channels=1
                torch.randn(2, 3, 8, 6),  # channels=3
                torch.randn(2, 1, 8, 6),  # channels=1
                torch.randn(2, 2, 8, 6),  # channels=2
            ),
            (
                torch.randn(2, 6, 8, 6),  # channels=6
                torch.randn(2, 1, 8, 6),  # channels=1
                torch.randn(2, 3, 8, 6),  # channels=3
                torch.randn(2, 2, 8, 6),  # channels=2
                torch.randn(2, 4, 8, 6),  # channels=4
                torch.randn(2, 1, 8, 6),  # channels=1
            ),
            (
                torch.randn(2, 2, 8, 6),  # channels=2
                torch.randn(2, 7, 8, 6),  # channels=7
                torch.randn(2, 1, 8, 6),  # channels=1
                torch.randn(2, 1, 8, 6),  # channels=1
                torch.randn(2, 3, 8, 6),  # channels=3
                torch.randn(2, 2, 8, 6),  # channels=2
            ),
        ]

        self.lower_module_and_test_output(
            cat_channels_module,
            sample_inputs,
            dynamic_shapes=dynamic_shapes,
            test_inputs=test_inputs,
        )

    def test_vulkan_backend_high_dimensional_tensors(self):
        class HighDimTensorModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # Unsqueeze inputs twice to create 5-dim tensors
                x_5d = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
                y_5d = torch.unsqueeze(torch.unsqueeze(y, 0), 0)
                # Add tensors together
                result = x_5d + y_5d
                return result

        high_dim_module = HighDimTensorModule()
        # Create 2 4-dim inputs
        sample_inputs = (
            torch.rand(size=(2, 3, 4, 5), dtype=torch.float32),
            torch.rand(size=(2, 3, 4, 5), dtype=torch.float32),
        )

        self.lower_module_and_test_output(high_dim_module, sample_inputs)

    def test_vulkan_backend_torchao_wo_quantized_linear(self):
        in_features = 1024
        out_features = 512
        bias = False
        group_size = 64
        weight_bits = 4

        class TorchAOQuantizedLinearModule(torch.nn.Module):
            def __init__(
                self,
                in_features: int,
                out_features: int,
                bias: bool = False,
                group_size: int = 64,
                weight_bits: int = 4,
            ):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
                self.group_size = group_size
                self.weight_bits = weight_bits

                if self.weight_bits == 4:
                    self.weight_dtype = torch.int4
                else:
                    self.weight_dtype = torch.int8

                self.quant_granularity = PerGroup(self.group_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

            def apply_quantization(self):
                """Apply TorchAO weight-only quantization to the linear layer."""
                q_config = IntxWeightOnlyConfig(
                    weight_dtype=self.weight_dtype,
                    granularity=self.quant_granularity,
                )
                quantize_(self, q_config)
                unwrap_tensor_subclass(self)
                return self

        # Test with GEMV pattern (batch_size=1, seq_len=1)
        quantized_linear_module = TorchAOQuantizedLinearModule(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            group_size=group_size,
            weight_bits=weight_bits,
        )

        # Apply quantization
        quantized_linear_module = quantized_linear_module.apply_quantization()

        # Test with 2D input (GEMV pattern)
        sample_inputs = (torch.randn(size=(1, in_features), dtype=torch.float32),)

        # Use higher tolerance since quantization introduces some error
        self.lower_module_and_test_output(
            quantized_linear_module, sample_inputs, atol=1e-2, rtol=1e-2
        )

        # Test with GEMM pattern (batch_size > 1)
        quantized_linear_module_gemm = TorchAOQuantizedLinearModule(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            group_size=group_size,
            weight_bits=weight_bits,
        )

        # Apply quantization
        quantized_linear_module_gemm = quantized_linear_module_gemm.apply_quantization()

        # Test with 3D input (GEMM pattern)
        sample_inputs_gemm = (
            torch.randn(size=(1, 248, in_features), dtype=torch.float32),
        )

        # Use higher tolerance since quantization introduces some error
        self.lower_module_and_test_output(
            quantized_linear_module_gemm, sample_inputs_gemm, atol=1e-2, rtol=1e-2
        )

    def test_vulkan_backend_xnnpack_pt2e_quantized_linear_sequence(self):
        """
        Test a sequence of linear layers quantized with XNNPACK quantization config.
        This test creates a module with multiple linear layers in sequence and applies
        XNNPACK symmetric quantization to test the quantized model execution.
        """

        import executorch.backends.vulkan.test.utils as test_utils

        class LinearSequenceModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(128, 64, bias=False)
                self.linear2 = torch.nn.Linear(64, 32, bias=False)
                self.linear3 = torch.nn.Linear(32, 16, bias=False)

                MAX = 0.75
                MIN = -0.25
                self.linear1.weight.data = test_utils.random_uniform_tensor(
                    self.linear1.weight.shape, MIN, MAX
                )
                self.linear2.weight.data = test_utils.random_uniform_tensor(
                    self.linear2.weight.shape, MIN, MAX
                )
                self.linear3.weight.data = test_utils.random_uniform_tensor(
                    self.linear3.weight.shape, MIN, MAX
                )

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        # Create the module
        linear_sequence_module = LinearSequenceModule()

        M = 32
        # Create sample inputs
        sample_inputs = (
            (
                test_utils.random_uniform_tensor(
                    (M, linear_sequence_module.linear1.in_features),
                    -0.25,
                    0.75,
                )
            ),
        )

        # Create XNNPACK quantizer with symmetric quantization config
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)

        # Test the quantized module using the existing quantize_and_lower_module function
        # Use higher tolerance since quantization introduces some error
        edge_program = quantize_and_lower_module(
            linear_sequence_module, sample_inputs, quantizer
        )

        et_program = edge_program.to_executorch()
        self.check_vk_delegation(et_program)

        self.run_delegated_model_and_check_output(
            et_program,
            linear_sequence_module,
            sample_inputs,
            atol=1e-2,
            rtol=1e-1,
        )

    def test_vulkan_backend_xnnpack_pt2e_quantized_conv_sequence(self):
        """
        Test a sequence of convolution layers quantized with PT2E quantization.
        This test creates a module with multiple Conv2d layers in sequence and applies
        XNNPACK symmetric quantization to test the quantized model execution.
        Similar to the linear sequence test but using convolution layers.
        """

        import executorch.backends.vulkan.test.utils as test_utils

        class ConvSequenceModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )

                MAX = 0.75
                MIN = -0.25
                self.conv1.weight.data = test_utils.random_uniform_tensor(
                    self.conv1.weight.shape, MIN, MAX
                )
                self.conv2.weight.data = test_utils.random_uniform_tensor(
                    self.conv2.weight.shape, MIN, MAX
                )
                self.conv3.weight.data = test_utils.random_uniform_tensor(
                    self.conv3.weight.shape, MIN, MAX
                )

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                return x

        # Create the module
        conv_sequence_module = ConvSequenceModule()

        input_tensor = test_utils.random_uniform_tensor(
            (1, 3, 32, 32),
            -0.25,
            0.75,
        )

        # Create sample inputs
        sample_inputs = (input_tensor,)

        # Create XNNPACK quantizer with symmetric quantization config
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)

        # Test the quantized module using the existing quantize_and_lower_module function
        # Use higher tolerance since quantization introduces some error
        edge_program = quantize_and_lower_module(
            conv_sequence_module, sample_inputs, quantizer
        )

        et_program = edge_program.to_executorch()
        self.check_vk_delegation(et_program)

        self.run_delegated_model_and_check_output(
            et_program,
            conv_sequence_module,
            sample_inputs,
            atol=1e-2,
            rtol=1e-1,
        )

    def test_vulkan_backend_xnnpack_pt2e_quantized_conv_sequence_all_reduced(self):
        """
        Test a sequence of convolution layers quantized with PT2E quantization.
        This test creates a module with multiple Conv2d layers in sequence and applies
        XNNPACK symmetric quantization to test the quantized model execution.
        Similar to the linear sequence test but using convolution layers.
        """

        import executorch.backends.vulkan.test.utils as test_utils

        class ConvSequenceModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=1,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )

                MAX = 0.75
                MIN = -0.25
                self.conv1.weight.data = test_utils.random_uniform_tensor(
                    self.conv1.weight.shape, MIN, MAX
                )
                self.conv2.weight.data = test_utils.random_uniform_tensor(
                    self.conv2.weight.shape, MIN, MAX
                )

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        # Create the module
        conv_sequence_module = ConvSequenceModule()

        input_tensor = test_utils.random_uniform_tensor(
            (1, 3, 32, 32),
            -0.25,
            0.75,
        )

        # Create sample inputs
        sample_inputs = (input_tensor,)

        # Create XNNPACK quantizer with symmetric quantization config
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)

        # Test the quantized module using the existing quantize_and_lower_module function
        # Use higher tolerance since quantization introduces some error
        edge_program = quantize_and_lower_module(
            conv_sequence_module, sample_inputs, quantizer
        )

        et_program = edge_program.to_executorch()
        self.check_vk_delegation(et_program)

        self.run_delegated_model_and_check_output(
            et_program,
            conv_sequence_module,
            sample_inputs,
            atol=1e-2,
            rtol=1e-1,
        )

    @unittest.skip("Cannot run on swiftshader due to no 8-bit int support")
    def test_vulkan_backend_torchao_8da4w_quantized_linear(self):
        """
        Test TorchAO 8da4w quantization (int8 dynamic activation + int4 weight) with Vulkan backend.
        This test uses the same quantization approach as the 8da4w qmode in quantize.py.
        """
        in_features = 1024
        out_features = 512
        bias = False
        group_size = 128

        class TorchAO8da4wQuantizedLinearModule(torch.nn.Module):
            def __init__(
                self,
                in_features: int,
                out_features: int,
                bias: bool = False,
                group_size: int = 128,
            ):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
                self.group_size = group_size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

            def apply_8da4w_quantization(self):
                """Apply TorchAO 8da4w quantization (int8 dynamic activation + int4 weight)."""
                from torchao.quantization import (
                    Int8DynamicActivationIntxWeightConfig,
                    quantize_,
                )
                from torchao.quantization.granularity import PerGroup
                from torchao.utils import unwrap_tensor_subclass

                quantize_(
                    self,
                    Int8DynamicActivationIntxWeightConfig(
                        weight_dtype=torch.int4, granularity=PerGroup(self.group_size)
                    ),
                )
                unwrap_tensor_subclass(self)
                return self

        # Test with GEMV pattern (batch_size=1, seq_len=1)
        quantized_linear_module = TorchAO8da4wQuantizedLinearModule(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            group_size=group_size,
        )

        # Apply 8da4w quantization
        quantized_linear_module = quantized_linear_module.apply_8da4w_quantization()

        # Test with 2D input (GEMV pattern)
        sample_inputs = (torch.randn(size=(1, in_features), dtype=torch.float32),)

        # Use higher tolerance since quantization introduces some error
        self.lower_module_and_test_output(
            quantized_linear_module, sample_inputs, atol=1e-2, rtol=1e-2
        )

        # Test with GEMM pattern (batch_size > 1)
        quantized_linear_module_gemm = TorchAO8da4wQuantizedLinearModule(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            group_size=group_size,
        )

        # Apply 8da4w quantization
        quantized_linear_module_gemm = (
            quantized_linear_module_gemm.apply_8da4w_quantization()
        )

        # Test with 3D input (GEMM pattern)
        sample_inputs_gemm = (
            torch.randn(size=(1, 248, in_features), dtype=torch.float32),
        )

        # Use higher tolerance since quantization introduces some error
        self.lower_module_and_test_output(
            quantized_linear_module_gemm, sample_inputs_gemm, atol=1e-2, rtol=1e-2
        )
