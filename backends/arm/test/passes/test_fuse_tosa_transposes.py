# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

from typing import cast, Dict, List, Protocol, Tuple

import torch
from executorch.backends.arm._passes import (
    AnnotateOutputDimOrderPass,
    FuseTosaTransposesPass,
    ToTosaMemoryFormatPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineINT,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass

input_t = Tuple[torch.Tensor]  # Input x


class ModuleMetadata(Protocol):
    ops_after_pass: Dict[str, int]
    ops_not_after_pass: List[str]

    def get_inputs(self) -> input_t: ...


class SingleConv2d(torch.nn.Module):
    """
    Test module with a single Conv2D.
    The TOSA pipeline inserts transposes at input/output boundaries.
    After FuseTosaTransposesPass, boundary transposes should remain.
    """

    # After pass: expect 2 boundary transposes (input, output)
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class ConvReluConv(torch.nn.Module):
    """
    Test module with Conv2D → ReLU → Conv2D pattern.
    This is the dominant pattern where intermediate transposes should cancel.
    After FuseTosaTransposesPass: only boundary transposes remain.
    """

    # After pass: expect 2 boundary transposes (input, output)
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class ConvChain(torch.nn.Module):
    """
    Test module with a chain of Conv2D ops.
    Multiple intermediate transpose pairs should cancel.
    """

    # After pass: expect 2 boundary transposes (input, output)
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class MaxPool2dChain(torch.nn.Module):
    """
    Test module with MaxPool2D operations that also use NHWC format.
    Transposes between pooling ops should cancel.
    """

    # After pass: expect 2 boundary transposes (input, output)
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(x)
        x = self.pool2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class NoOpsRequiringNHWC(torch.nn.Module):
    """
    Test module with no operations requiring NHWC format (just element-wise ops).
    Should have input/output boundary transposes only.
    """

    # After pass: expect 2 boundary transposes (input, output)
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        x = x * 2
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class TransposeWithMultipleUsers(torch.nn.Module):
    """
    Test module where a conv output is used by multiple consumers.
    Tests that transposes are properly handled with fan-out.
    """

    # After pass: transposes depend on how fan-out is handled
    ops_after_pass: Dict[str, int] = {}  # Don't check count, just verify it runs
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # x is used by both conv2 and the residual addition
        y = self.conv2(x)
        return y + x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


modules: Dict[str, ModuleMetadata] = {
    "single_conv2d": SingleConv2d(),
    "conv_relu_conv": ConvReluConv(),
    "conv_chain": ConvChain(),
    "maxpool2d_chain": MaxPool2dChain(),
    "no_ops_requiring_nhwc": NoOpsRequiringNHWC(),
    "transpose_with_multiple_users": TransposeWithMultipleUsers(),
}


@common.parametrize("module", modules)
def test_fuse_tosa_transposes_pass(module: ModuleMetadata) -> None:
    """
    Test FuseTosaTransposesPass with explicit pass pipeline.
    Verifies that redundant transposes are eliminated.
    """
    module_nn = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        module_nn,
        module.get_inputs(),
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        pass_list=[RemoveGetItemPass, AnnotateOutputDimOrderPass],
        passes_with_exported_program=[
            ToTosaMemoryFormatPass,
            FuseTosaTransposesPass,
        ],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # Eager execution not possible after TOSA transposes
    pipeline.run()


@common.parametrize("module", modules)
def test_fuse_tosa_transposes_functional(module: ModuleMetadata) -> None:
    """
    Test FuseTosaTransposesPass with full TOSA pipeline.
    Verifies functional correctness end-to-end.
    """
    module_nn = cast(torch.nn.Module, module)
    pipeline = TosaPipelineINT[input_t](module_nn, module.get_inputs(), [])
    pipeline.run()


def test_identity_transpose_elimination() -> None:
    """
    Test that identity transposes are eliminated.
    Uses a simple pass-through module.
    """

    class IdentityModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def get_inputs(self) -> input_t:
            return (torch.rand(1, 16, 8, 8),)

    module = IdentityModule()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        pass_list=[RemoveGetItemPass, AnnotateOutputDimOrderPass],
        passes_with_exported_program=[
            ToTosaMemoryFormatPass,
            FuseTosaTransposesPass,
        ],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


# ------------------------------------------------------------------
# Tests for Pattern 4: Propagation through elementwise ops
# ------------------------------------------------------------------


class ConvBnReLUConv(torch.nn.Module):
    """
    Test module: Conv2d → BatchNorm → ReLU → Conv2d.
    BatchNorm decomposes to mul/add (elementwise ops).
    FuseTosaTransposesPass should propagate transposes through
    the elementwise chain between the two convolutions.
    """

    # After pass: only boundary transposes should remain
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class ConvAddScalarConv(torch.nn.Module):
    """
    Test module: Conv2d → add(scalar) → Conv2d.
    The scalar add is a binary elementwise op with broadcast-safe operand.
    FuseTosaTransposesPass should propagate through it.
    """

    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x + 1.0  # scalar add — safe for transpose propagation
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class ConvClampConv(torch.nn.Module):
    """
    Test module: Conv2d → clamp → Conv2d.
    Clamp is a unary elementwise op (with scalar bounds).
    FuseTosaTransposesPass should propagate through it.
    """

    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class ConvSigmoidConv(torch.nn.Module):
    """
    Test module: Conv2d → sigmoid → Conv2d.
    Sigmoid is a unary elementwise op (TOSA TABLE).
    FuseTosaTransposesPass should propagate through it.
    """

    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


class ConvTanhConv(torch.nn.Module):
    """
    Test module: Conv2d → tanh → Conv2d.
    Tanh is a unary elementwise op (TOSA TABLE).
    FuseTosaTransposesPass should propagate through it.
    """

    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 16, 8, 8),)


propagation_modules: Dict[str, ModuleMetadata] = {
    "conv_bn_relu_conv": ConvBnReLUConv(),
    "conv_add_scalar_conv": ConvAddScalarConv(),
    "conv_clamp_conv": ConvClampConv(),
    "conv_sigmoid_conv": ConvSigmoidConv(),
    "conv_tanh_conv": ConvTanhConv(),
}


@common.parametrize("module", propagation_modules)
def test_propagation_through_elementwise(module: ModuleMetadata) -> None:
    """
    Test FuseTosaTransposesPass Pattern 4: propagation through elementwise ops.
    Verifies that transposes separated by elementwise ops are cancelled.
    """
    module_nn = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        module_nn,
        module.get_inputs(),
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        pass_list=[RemoveGetItemPass, AnnotateOutputDimOrderPass],
        passes_with_exported_program=[
            ToTosaMemoryFormatPass,
            FuseTosaTransposesPass,
        ],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.parametrize("module", propagation_modules)
def test_propagation_functional(module: ModuleMetadata) -> None:
    """
    Test FuseTosaTransposesPass Pattern 4 with full TOSA pipeline.
    Verifies functional correctness end-to-end.
    """
    module_nn = cast(torch.nn.Module, module)
    pipeline = TosaPipelineINT[input_t](module_nn, module.get_inputs(), [])
    pipeline.run()
