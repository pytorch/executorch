# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import torch
import torch.fx

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

input_single = Tuple[torch.Tensor]
input_double = Tuple[torch.Tensor, torch.Tensor]


class WhileTwoInputsTwoOutputs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, lhs: torch.Tensor, rhs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def cond_fn(lhs_val: torch.Tensor, rhs_val: torch.Tensor) -> torch.Tensor:
            total = torch.sum(rhs_val)
            zero = torch.zeros_like(total)
            return torch.gt(total, zero).squeeze()

        def body_fn(
            lhs_val: torch.Tensor, rhs_val: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            next_lhs = torch.add(lhs_val, rhs_val)
            next_rhs = torch.sub(rhs_val, torch.full((1,), 1.0))
            return (next_lhs, next_rhs)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (lhs, rhs),
            (),
        )
        return result  # type: ignore


class WhileOneInputOneBufferTwoOutputs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((30.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.lt(total, limit).squeeze()

        def body_fn(
            value: torch.Tensor, limit: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return (torch.add(value, value), limit.clone())

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value, self.threshold),
            (),
        )
        return result  # type: ignore


class WhileAdditionalArg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((30.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.lt(total, limit).squeeze()

        def body_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            return torch.add(value, value)

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value,),
            (self.threshold,),
        )
        return result  # type: ignore


class WhileSingleCapturedOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor((30.0,)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        def cond_fn(value: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
            total = value.sum()
            return torch.lt(total, limit).squeeze()

        def body_fn(
            value: torch.Tensor, limit: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return (torch.add(value, value), limit.clone())

        result = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (value, self.threshold),
            (),
        )
        return result[0]  # type: ignore


def _single_input_case(
    module_factory: Callable[[], torch.nn.Module],
) -> Callable[[], Tuple[torch.nn.Module, input_single]]:
    def _create() -> Tuple[torch.nn.Module, input_single]:
        return module_factory(), (torch.ones(2, 3),)

    return _create


def _dual_input_case(
    module_factory: Callable[[], torch.nn.Module],
) -> Callable[[], Tuple[torch.nn.Module, input_double]]:
    def _create() -> Tuple[torch.nn.Module, input_double]:
        return module_factory(), (torch.zeros(2, 3), torch.full((2, 3), -2.0))

    return _create


test_cases: dict[str, Callable[[], Tuple[torch.nn.Module, Tuple]]] = {
    "two_in_two_out": _dual_input_case(WhileTwoInputsTwoOutputs),
    "one_in_one_buffer_two_out": _single_input_case(WhileOneInputOneBufferTwoOutputs),
    "additional_arg": _single_input_case(WhileAdditionalArg),
    "two_in_one_captured_out": _single_input_case(WhileSingleCapturedOutput),
}


@common.parametrize(
    "case",
    test_cases,
    xfails={
        "additional_arg": "Support not implemented.",
        "two_in_one_captured_out": "When only one output is used, the second one is removed, which is not allowed in TOSA.",
    },
)
def test_while_loop_tosa_FP(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    pipeline = TosaPipelineFP[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
        tosa_extensions=["cf"],
    )
    pipeline.run()


@common.parametrize(
    "case",
    test_cases,
    xfails={
        "additional_arg": "Support not implemented.",
        "two_in_one_captured_out": "When only one output is used, the second one is removed, which is not allowed in TOSA.",
    },
)
def test_while_loop_tosa_INT(case: Callable[[], Tuple[torch.nn.Module, Tuple]]):
    module, example_inputs = case()
    pipeline = TosaPipelineINT[tuple](
        module,
        example_inputs,
        "torch.ops.higher_order.while_loop",
        tosa_extensions=["cf"],
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        ArmTester.check_not,
        pipeline.tester,
        ["torch.ops.higher_order.while_loop"],
    )
    pipeline.run()
