# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from pytest import mark

aten_op = "torch.ops.higher_order.cond"
exir_op = "torch.ops.higher_order.cond"

input_t1 = Tuple[torch.Tensor]
input_t2 = Tuple[torch.Tensor, torch.Tensor]


class CondZeroArgsOneOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_branch() -> torch.Tensor:
            return torch.zeros(10)

        def false_branch() -> torch.Tensor:
            return torch.ones(10)

        predicate = x.sum() > 0
        return torch.cond(predicate, true_branch, false_branch, [])


class CondOneArgOneOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_branch(arg: torch.Tensor) -> torch.Tensor:
            return torch.sin(arg)

        def false_branch(arg: torch.Tensor) -> torch.Tensor:
            return torch.cos(arg)

        predicate = x.sum() > 0
        return torch.cond(predicate, true_branch, false_branch, [x])


class CondOneArgBufferOneOutput(torch.nn.Module):
    def __init__(self, *args: common.Any, **kwargs: common.Any) -> None:
        super().__init__(*args, **kwargs)
        self.buffer = torch.rand(1, 1, 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_branch(arg: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
            return torch.sin(arg) + buffer

        def false_branch(arg: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
            return torch.cos(arg) + buffer

        predicate = x.sum() > 0
        return torch.cond(predicate, true_branch, false_branch, [x, self.buffer])


class CondOneArgAndScalarOneOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_branch(arg: torch.Tensor) -> torch.Tensor:
            return arg + 1.0

        def false_branch(arg: torch.Tensor) -> torch.Tensor:
            return arg - 1.0

        predicate = x.sum() > 0
        return torch.cond(predicate, true_branch, false_branch, [x])


class CondOneArgTwoOutputs(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def true_branch(arg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return arg + torch.sin(arg), arg - torch.sin(arg)

        def false_branch(arg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return arg - torch.cos(arg), arg + torch.cos(arg)

        predicate = x.flatten().sum() > 0
        return torch.cond(predicate, true_branch, false_branch, [x])


class CondNestedOneArgOneOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def inner_true(arg: torch.Tensor) -> torch.Tensor:
            return arg + torch.full((1,), (1.0))

        def inner_false(arg: torch.Tensor) -> torch.Tensor:
            return arg - torch.full((1,), (1.0))

        def outer_true(arg: torch.Tensor) -> torch.Tensor:
            inner_predicate = arg.mean() > 0
            return torch.cond(inner_predicate, inner_true, inner_false, [arg])

        def outer_false(arg: torch.Tensor) -> torch.Tensor:
            return arg * torch.full((1,), (1.0))

        predicate = x.sum() > 0
        return torch.cond(predicate, outer_true, outer_false, [x])


class CondMultipleOneArgOneOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def first_true(arg: torch.Tensor) -> torch.Tensor:
            return arg.sigmoid()

        def first_false(arg: torch.Tensor) -> torch.Tensor:
            return arg.relu()

        first_predicate = x.sum() > 0
        intermediate = torch.cond(first_predicate, first_true, first_false, [x])

        def second_true(arg: torch.Tensor) -> torch.Tensor:
            return arg.sin()

        def second_false(arg: torch.Tensor) -> torch.Tensor:
            return arg.cos()

        second_predicate = intermediate.mean() > 0
        return torch.cond(second_predicate, second_true, second_false, [intermediate])


class CondTwoArgsOneOutput(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        def true_branch(arg_l: torch.Tensor, arg_r: torch.Tensor) -> torch.Tensor:
            return arg_l + arg_r

        def false_branch(arg_l: torch.Tensor, arg_r: torch.Tensor) -> torch.Tensor:
            return arg_l - arg_r

        predicate = (lhs - rhs).sum() > 0
        return torch.cond(predicate, true_branch, false_branch, [lhs, rhs])


class CondTwoArgsTwoOutputs(torch.nn.Module):
    def forward(
        self, lhs: torch.Tensor, rhs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def true_branch(
            arg_l: torch.Tensor, arg_r: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return arg_l + arg_r, arg_l * arg_r

        def false_branch(
            arg_l: torch.Tensor, arg_r: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            diff = arg_l - arg_r
            return diff, arg_l + diff

        predicate = (lhs * rhs).sum() > 0
        return torch.cond(predicate, true_branch, false_branch, [lhs, rhs])


class CondConv1d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.true_conv = torch.nn.Conv1d(2, 4, 3, padding=1)
        self.false_conv = torch.nn.Conv1d(2, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_branch(arg: torch.Tensor) -> torch.Tensor:
            return self.true_conv(arg)

        def false_branch(arg: torch.Tensor) -> torch.Tensor:
            return self.false_conv(arg)

        return torch.cond(x.sum() > 0, true_branch, false_branch, [x])


class NestedCondConv1d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner_true_conv = torch.nn.Conv1d(2, 4, 3, padding=1)
        self.inner_false_conv = torch.nn.Conv1d(2, 4, 3, padding=1)
        self.outer_false_conv = torch.nn.Conv1d(2, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def outer_true(arg: torch.Tensor) -> torch.Tensor:
            def inner_true(inner_arg: torch.Tensor) -> torch.Tensor:
                return self.inner_true_conv(inner_arg)

            def inner_false(inner_arg: torch.Tensor) -> torch.Tensor:
                return self.inner_false_conv(inner_arg)

            return torch.cond(
                arg.mean() > 0.5,
                inner_true,
                inner_false,
                [arg],
            )

        def outer_false(arg: torch.Tensor) -> torch.Tensor:
            return self.outer_false_conv(arg)

        return torch.cond(x.sum() > 0, outer_true, outer_false, [x])


def _single_input_case(
    module_factory: Callable[[], torch.nn.Module]
) -> Callable[[], tuple[torch.nn.Module, input_t1]]:
    def _create() -> tuple[torch.nn.Module, input_t1]:
        return module_factory(), (torch.randn(1, 1, 2, 2),)

    return _create


def _dual_input_case(
    module_factory: Callable[[], torch.nn.Module]
) -> Callable[[], tuple[torch.nn.Module, input_t2]]:
    def _create() -> tuple[torch.nn.Module, input_t2]:
        return module_factory(), (torch.randn(2, 3, 4, 6), torch.randn(2, 3, 4, 6))

    return _create


test_cases: dict[str, Callable[[], tuple[torch.nn.Module, tuple]]] = {
    "zero_args_one_output": _single_input_case(CondZeroArgsOneOutput),
    "one_arg_one_output": _single_input_case(CondOneArgOneOutput),
    "one_arg_const_one_output": _single_input_case(CondOneArgBufferOneOutput),
    "one_arg_and_scalar_one_output": _single_input_case(CondOneArgAndScalarOneOutput),
    "one_arg_two_outputs": _single_input_case(CondOneArgTwoOutputs),
    "two_args_one_output": _dual_input_case(CondTwoArgsOneOutput),
    "two_args_two_outputs": _dual_input_case(CondTwoArgsTwoOutputs),
    "nested_one_arg_one_output": _single_input_case(CondNestedOneArgOneOutput),
    "multiple_one_arg_one_output": _single_input_case(CondMultipleOneArgOneOutput),
}


def _make_calibration_samples(
    module: torch.nn.Module, example_inputs: tuple
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Return one example input that triggers the if branch, and one that
    triggers the else branch.
    """

    if isinstance(module, CondTwoArgsOneOutput):
        # Predicate is sum(lhs-rhs) > 0
        lhs, rhs = example_inputs
        if_example_inputs = (lhs, rhs)
        else_example_inputs = (rhs, lhs)
    elif isinstance(module, CondTwoArgsTwoOutputs):
        # Predicate is sum(lhs*rhs) > 0
        lhs, rhs = example_inputs
        if_example_inputs = (lhs, rhs)
        else_example_inputs = (lhs, -rhs)
    else:
        # Predicate is sum(x) > 0
        (x,) = example_inputs
        if_example_inputs = (x,)
        else_example_inputs = (-x,)

    return (if_example_inputs, else_example_inputs)


def _set_branch_calibration_samples(
    pipeline, module: torch.nn.Module, example_inputs: tuple
) -> None:
    calibration_samples = _make_calibration_samples(module, example_inputs)
    quant_stage_pos = pipeline.find_pos("quantize")
    quant_stage = pipeline._stages[quant_stage_pos].args[0]
    quant_stage.calibration_samples = calibration_samples


@common.parametrize("case", test_cases)
def test_cond_tosa_FP(case: Callable[[], tuple[torch.nn.Module, tuple]]):
    module, example_inputs = case()
    pipeline = TosaPipelineFP[tuple](
        module, example_inputs, aten_op, tosa_extensions=["cf"]
    )

    # Make sure no cond ops are left after partitioning.
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        ArmTester.check_not,
        pipeline.tester,
        ["torch.ops.higher_order.cond"],
    )
    pipeline.run()


def test_cond_conv1d_tosa_FP():
    TosaPipelineFP[tuple](
        CondConv1d(),
        (torch.randn(1, 2, 8),),
        aten_op,
        tosa_extensions=["cf"],
    ).run()


def test_cond_conv1d_tosa_INT():
    # Regression test for a quantized Conv1d whose weight and bias are captured
    # by a branch. RewriteConvPass must resolve them in the enclosing graph.
    module = CondConv1d()
    example_inputs = (torch.randn(1, 2, 8),)
    pipeline = TosaPipelineINT[tuple](
        module,
        example_inputs,
        aten_op,
        tosa_extensions=["cf"],
    )
    _set_branch_calibration_samples(pipeline, module, example_inputs)
    pipeline.run()


def test_cond_conv1d_tosa_INT_branches():
    # Exercise both branch graphs explicitly. A single random input only checks
    # the selected branch at runtime even though both branches are lowered.
    for example_input in (torch.rand(1, 2, 8), -torch.rand(1, 2, 8)):
        module = CondConv1d()
        example_inputs = (example_input,)
        pipeline = TosaPipelineINT[tuple](
            module,
            example_inputs,
            aten_op,
            tosa_extensions=["cf"],
        )
        _set_branch_calibration_samples(pipeline, module, example_inputs)
        pipeline.run()


def test_nested_cond_conv1d_tosa_INT():
    # These samples select inner-true, inner-false, and outer-false respectively.
    # They prove that constant provenance and rescale chains survive more than
    # one control-flow boundary, not only a single torch.cond.
    calibration_samples = (
        (torch.ones(1, 2, 8),),
        (torch.full((1, 2, 8), 0.1),),
        (-torch.ones(1, 2, 8),),
    )
    for example_inputs in calibration_samples:
        pipeline = TosaPipelineINT[tuple](
            NestedCondConv1d(),
            example_inputs,
            aten_op,
            tosa_extensions=["cf"],
        )
        quant_stage_pos = pipeline.find_pos("quantize")
        quant_stage = pipeline._stages[quant_stage_pos].args[0]
        # Calibrate all paths for every runtime case; otherwise the result could
        # fail because an unselected branch lacks quantization observations.
        quant_stage.calibration_samples = calibration_samples
        pipeline.run()


@common.parametrize("case", test_cases)
def test_cond_tosa_INT(case: Callable[[], tuple[torch.nn.Module, tuple]]):
    module, example_inputs = case()
    pipeline = TosaPipelineINT[tuple](
        module,
        example_inputs,
        aten_op,
        tosa_extensions=["cf"],
    )
    _set_branch_calibration_samples(pipeline, module, example_inputs)
    # Make sure no cond ops are left after partitioning.
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        ArmTester.check_not,
        pipeline.tester,
        ["torch.ops.higher_order.cond"],
    )
    pipeline.run()


@common.parametrize("case", test_cases)
def test_cond_u55_INT(case: Callable[[], tuple[torch.nn.Module, tuple]]):
    module, example_inputs = case()
    pipeline = OpNotSupportedPipeline[tuple](module, example_inputs, {aten_op: 1})
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.parametrize("case", test_cases)
@common.XfailIfNoCorstone320.with_args(raises=None)
def test_cond_u85_INT(case: Callable[[], tuple[torch.nn.Module, tuple]]):
    module, example_inputs = case()
    pipeline = EthosU85PipelineINT[tuple](module, example_inputs, aten_op, exir_op)
    _set_branch_calibration_samples(pipeline, module, example_inputs)
    pipeline.run()


@mark.skip("Cond not supported in model_converter.")
@common.parametrize(
    "case",
    test_cases,
)
@common.SkipIfNoModelConverter
def test_cond_vgf_FP(case: Callable[[], tuple[torch.nn.Module, tuple]]):
    module, example_inputs = case()
    VgfPipeline[tuple](
        module,
        example_inputs,
        aten_op,
        exir_op,
    ).run()


@mark.skip("Cond not supported in model_converter.")
@common.parametrize(
    "case",
    test_cases,
)
@common.SkipIfNoModelConverter
def test_cond_vgf_INT(case: Callable[[], tuple[torch.nn.Module, tuple]]):
    module, example_inputs = case()
    pipeline = VgfPipeline[tuple](
        module,
        example_inputs,
        aten_op,
        exir_op,
    )
    _set_branch_calibration_samples(pipeline, module, example_inputs)
    pipeline.run()
