# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, cast, ClassVar, Dict, Protocol, Tuple, TypeVar

import torch
from executorch.backends.arm._passes.fuse_equal_placeholders_pass import (
    FuseEqualPlaceholdersPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineFP,
)

input_t = Tuple[torch.Tensor]  # Input x


class ModuleWithEqualPlaceholderAttrs(Protocol):
    ops_before_pass: Dict[str, int]
    ops_after_pass: Dict[str, int]
    ops_not_after_pass: list[str]

    def get_inputs(self) -> input_t: ...


T = TypeVar("T")
TestDecorator = Callable[[Callable[[T], None]], Callable[[T], None]]


def _typed_parametrize(test_data: Dict[str, T]) -> TestDecorator:
    return cast(TestDecorator, common.parametrize("module", test_data))


class FuseWeightsConstants(torch.nn.Module):
    ops_before_pass: ClassVar[Dict[str, int]] = {}
    ops_after_pass: ClassVar[Dict[str, int]] = {}
    ops_not_after_pass: ClassVar[list[str]] = []

    def __init__(
        self,
    ):
        super().__init__()
        self.weights1 = torch.rand(1, 2, 1)
        self.weights2 = deepcopy(self.weights1)
        self.bias1 = torch.rand(1)
        self.bias2 = deepcopy(self.bias1)
        self.bias3 = deepcopy(self.bias1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.conv1d(x, self.weights1, self.bias1)
            + torch.conv1d(x, self.weights2, self.bias2)
            + self.bias3
        )

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 2, 8),)


class FuseWeightsStateDict(torch.nn.Module):
    ops_before_pass: ClassVar[Dict[str, int]] = {}
    ops_after_pass: ClassVar[Dict[str, int]] = {}
    ops_not_after_pass: ClassVar[list[str]] = []

    def __init__(
        self,
    ):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=8, out_features=2, bias=True)
        self.fc2 = deepcopy(self.fc1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x) + self.fc2(x)

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 2, 8),)


class NotFuseTensorWithDifferentType(torch.nn.Module):

    ops_before_pass: ClassVar[Dict[str, int]] = {}
    ops_after_pass: ClassVar[Dict[str, int]] = {}
    ops_not_after_pass: ClassVar[list[str]] = []

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: A float tensor (dtype=torch.float32)
            y: An int tensor (dtype=torch.int32)
        """
        a = torch.tensor(1.0, dtype=torch.float32)
        b = torch.tensor(1, dtype=torch.int32)
        m = x < a
        n = y > b
        return m, n


constants_modules: Dict[str, ModuleWithEqualPlaceholderAttrs] = {
    "fuse_constants": cast(ModuleWithEqualPlaceholderAttrs, FuseWeightsConstants()),
}

parametrize_constants = _typed_parametrize(constants_modules)


@parametrize_constants
def test_fuse_equal_placeholders_tosa_FP_constants(
    module: ModuleWithEqualPlaceholderAttrs,
) -> None:
    pipeline = PassPipeline[input_t](
        cast(torch.nn.Module, module),
        module.get_inputs(),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[FuseEqualPlaceholdersPass],
    )
    pipeline.run()

    # Check that weights and bias has been merged.
    exp_program = pipeline.tester.get_artifact().exported_program()
    constant_keys = list(exp_program.constants.keys())
    assert len(constant_keys) == 2, "FuseEqualPlaceholders constants failed"
    assert "_common" in constant_keys[0], "FuseEqualPlaceholders constants failed"
    assert "_common" in constant_keys[1], "FuseEqualPlaceholders constants failed"


def test_fuse_equal_placeholders_tosa_FP_state_dict() -> None:
    module = FuseWeightsStateDict()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        passes_with_exported_program=[FuseEqualPlaceholdersPass],
    )
    pipeline.run()

    # Check that weights and bias has been merged.
    exp_program = pipeline.tester.get_artifact().exported_program()
    state_dict_keys = list(exp_program.state_dict.keys())
    assert len(state_dict_keys) == 2, "FuseEqualPlaceholders state_dict failed"
    assert "_common" in state_dict_keys[0], "FuseEqualPlaceholders state_dict failed"
    assert "_common" in state_dict_keys[1], "FuseEqualPlaceholders state_dict failed"


def test_fuse_equal_placeholders_tosa_FP_no_fuse_type_diff():
    module = NotFuseTensorWithDifferentType()
    data = (
        torch.rand(
            1,
        ),
        torch.randint(
            0,
            10,
            (1,),
            dtype=torch.int,
        ),
    )
    pipeline = TosaPipelineFP[input_t](
        module,
        data,
        aten_op=[],
    )
    pipeline.run()
