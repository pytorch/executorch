# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# This file contains logic to run generated operator tests using the FACTO
# library (https://github.com/meta-pytorch/FACTO). To run the tests, first
# clone and install FACTO by running pip install . from the FACTO source
# directory. Then, from the executorch root directory, run the following:
#
# python -m unittest backends.test.facto.test_facto.FactoTestsXNNPACK
#
# Useful environment variables:
# FACTO_OPS="abs.default,acos.default" limits generated tests to selected ops.
# FACTO_MAX_CASES=10 limits the number of generated cases per op.
#

import copy
import fnmatch
import functools
import os
import traceback
import unittest
from typing import Any, Callable, Sequence

import torch
from executorch.backends.test.harness.tester import Tester as BackendTester
from executorch.backends.xnnpack.test.tester.tester import Tester as XnnpackTester

try:
    from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
    from facto.inputgen.specs.model import ConstraintProducer as cp, Spec
    from facto.inputgen.utils.random_manager import random_manager
    from facto.specdb.db import SpecDictDB
except ImportError as exc:
    raise ImportError(
        "FACTO is required to run generated operator tests. Install it with "
        "`pip install -e backends/cadence/utils/FACTO`."
    ) from exc
from torch._ops import OpOverload

from .facto_specs import ExtraSpecDB

CombinedSpecDB = SpecDictDB | ExtraSpecDB

COMMON_TENSOR_CONSTRAINTS = [
    cp.Rank.Ge(lambda deps: 1),  # Avoid zero and high rank tensors.
    cp.Rank.Le(lambda deps: 4),
    cp.Size.Ge(lambda deps, r, d: 1),  # Keep sizes reasonable.
    cp.Size.Le(lambda deps, r, d: 2**9),
]

COMMON_SCALAR_CONSTRAINS = [
    cp.Value.Ge(lambda deps, dtype: -1000),
    cp.Value.Le(lambda deps, dtype: 1000),
]

# Operator args are treated as runtime graph inputs if the argument name is
# in this list.
RUNTIME_INPUT_NAMES = {
    "self",
    "tensor",
    "other",
}


TesterFactory = Callable[[torch.nn.Module, tuple[Any, ...]], BackendTester]


def _facto_max_cases() -> int | None:
    max_cases = os.environ.get("FACTO_MAX_CASES")
    if max_cases is None:
        return None
    return int(max_cases)


def _selected_op_names() -> list[str]:
    op_patterns = os.environ.get("FACTO_OPS")
    if op_patterns is None or not op_patterns.strip():
        return list(CombinedSpecDB.keys())

    patterns = [pattern.strip() for pattern in op_patterns.split(",")]
    patterns = [pattern for pattern in patterns if pattern]
    return [
        op_name
        for op_name in CombinedSpecDB.keys()
        if any(fnmatch.fnmatch(op_name, pattern) for pattern in patterns)
    ]


def _patch_spec(spec: Spec) -> Spec:
    spec = copy.deepcopy(spec)
    for inspec in spec.inspec:
        if inspec.type.is_tensor():
            inspec.constraints.extend(COMMON_TENSOR_CONSTRAINTS)
        elif inspec.type.is_scalar():
            inspec.constraints.extend(COMMON_SCALAR_CONSTRAINS)
    return spec


class OpModel(torch.nn.Module):
    """
    Wraps a single torch operator in an nn.Module.
    """

    def __init__(
        self,
        op: OpOverload,
        runtime_input_count: int,
        fixed_args: Sequence[Any],
        fixed_kwargs: dict[str, Any],
    ):
        super().__init__()
        self.op = op
        self.runtime_input_count = runtime_input_count
        self.fixed_kwargs = fixed_kwargs

        # Register parameters for fixed tensors. Some things will choke on
        # constant tensor weights, for example.
        new_args = []
        for i, arg in enumerate(fixed_args):
            if isinstance(arg, torch.Tensor):
                param = torch.nn.Parameter(arg, requires_grad=False)
                param_name = f"arg_{i}_param"
                setattr(self, param_name, param)
                self.register_parameter(param_name, param)
                new_args.append(param)
            else:
                new_args.append(arg)
        self.fixed_args = tuple(new_args)

    def forward(self, *args, **kwargs):
        return self.op(*(args + self.fixed_args), **(kwargs | self.fixed_kwargs))


# The convolution model has some minor wrapper logic around the actual convolution
# operator. Most of the backends are expecting this form.
# TODO (gjcomer) Investigate these discrepencies.
class ConvModel(OpModel):
    def forward(self, *args, **kwargs):
        weight, bias, stride, padding, dilation, transposed, output_padding, groups = (
            self.fixed_args
        )

        if not transposed:
            if len(weight.shape) == 3:
                op = torch.nn.functional.conv1d
            elif len(weight.shape) == 4:
                op = torch.nn.functional.conv2d
            elif len(weight.shape) == 5:
                op = torch.nn.functional.conv3d

            return op(args[0], weight, bias, stride, padding, dilation, groups)
        else:
            if len(weight.shape) == 3:
                op = torch.nn.functional.conv_transpose1d
            elif len(weight.shape) == 4:
                op = torch.nn.functional.conv_transpose2d
            elif len(weight.shape) == 5:
                op = torch.nn.functional.conv_transpose3d

            return op(
                args[0], weight, bias, stride, padding, output_padding, groups, dilation
            )


def get_module_for_op(op: OpOverload):
    if op == torch.ops.aten.convolution.default:
        return ConvModel
    else:
        return OpModel


class FactoTestsBase(unittest.TestCase):
    __test__ = False

    def __init__(self, tester_factory: TesterFactory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tester_factory = tester_factory

    @classmethod
    def _generate_test(cls, op_name: str) -> None:
        # Find the torch op with the given name.
        sections = op_name.split(".")
        torch_op = functools.reduce(getattr, sections, torch.ops.aten)

        test_name = "test_" + op_name.replace(".", "_")

        def test_body(self):
            self._test_op(torch_op)

        setattr(cls, test_name, test_body)

    @staticmethod
    def get_runtime_input_count(spec: Spec):
        # Determine which inputs are fixed at tracing time (weights, for example),
        # vs inputs to the runtime graph. We currently assume that the runtime graph
        # inputs start at the beginning of the arg list and are contiguous.
        #
        # Args are consider to be runtime inputs if they are positional and are named
        # one of RUNTIME_INPUT_NAMES. If none match, we assume only the first arg is a
        # runtime input.
        runtime_input_count = 0
        for inspec in spec.inspec:
            is_runtime_input = (
                inspec.type.is_tensor() and inspec.name.lower() in RUNTIME_INPUT_NAMES
            )
            if is_runtime_input:
                runtime_input_count += 1
            else:
                break

        return max(1, runtime_input_count)

    def setUp(self):
        torch.set_printoptions(threshold=3)

    def _patch_spec_for_backend(self, spec: Spec, op_name: str) -> Spec:
        return spec

    def _should_skip_case(self, posargs: Sequence[Any]) -> bool:
        return False

    def _run_delegated_case(self, tester: BackendTester) -> None:
        tester.to_executorch().serialize().run_method_and_compare_outputs()

    def _should_fail_on_failures(self) -> bool:
        return False

    def _count_as_test_failures(
        self,
        *,
        eager_fail_count: int,
        export_fail_count: int,
        fail_count: int,
    ) -> int:
        return fail_count

    def _test_op(self, op: OpOverload) -> None:  # noqa: C901
        random_manager.seed(0)

        # Strip namespace
        op_name = op.name().split("::")[-1]

        # Default to .default overload
        if "." not in op_name:
            op_name += ".default"

        # Find and patch op spec
        if op_name not in CombinedSpecDB:
            raise ValueError(f"Operator {op_name} not found in SpecDictDB.")
        spec = _patch_spec(CombinedSpecDB[op_name])
        spec = self._patch_spec_for_backend(spec, op_name)

        runtime_input_count = FactoTestsBase.get_runtime_input_count(spec)

        print(f"Op: {op_name}, {runtime_input_count} runtime inputs")

        # Run test cases
        eager_fail_count = 0
        export_fail_count = 0
        success_count_delegated = 0
        success_count_undelegated = 0
        fail_count = 0

        max_cases = _facto_max_cases()
        for case_index, (posargs, inkwargs, _) in enumerate(
            ArgumentTupleGenerator(spec).gen()
        ):
            if max_cases is not None and case_index >= max_cases:
                break

            try:
                if self._should_skip_case(posargs):
                    continue

                module_cls = get_module_for_op(op)
                model = module_cls(
                    op, runtime_input_count, posargs[runtime_input_count:], inkwargs
                )

                # Sanity check to make sure it runs in eager. This can present nicer error
                # messages sometimes compared to tracing.
                try:
                    model(*posargs[:runtime_input_count])
                except Exception as e:
                    eager_fail_count += 1
                    print(f"Eager execution failed: {e}")
                    continue

                tester = self._tester_factory(
                    model, tuple(posargs[:runtime_input_count])
                )

                # Dynamo will also fail to handle some patterns that are valid in eager.
                try:
                    tester.export()
                except Exception:
                    export_fail_count += 1
                    print("Export failed.")
                    continue

                tester.to_edge_transform_and_lower()

                is_delegated = any(
                    n.target == torch._higher_order_ops.executorch_call_delegate
                    for n in tester.stages[tester.cur].graph_module.graph.nodes
                    if n.op == "call_function"
                )

                # Only run the runtime test if the op was delegated.
                if is_delegated:
                    self._run_delegated_case(tester)

                if is_delegated:
                    success_count_delegated += 1
                else:
                    success_count_undelegated += 1
            except Exception:
                fail_count += 1
                print("Args:")
                for arg in posargs:
                    if isinstance(arg, torch.Tensor):
                        print(f"  {arg.dtype} {arg.shape}")
                    else:
                        print(f"  {arg}")

                traceback.print_exc()

        print(
            f"{success_count_delegated + success_count_undelegated} PASS, {fail_count} FAIL"
        )
        print(
            f"  {success_count_delegated} DELEGATED, {success_count_undelegated} UNDELEGATED"
        )
        print(f"  {eager_fail_count} EAGER_FAILED, {export_fail_count} EXPORT_FAILED")

        counted_failures = self._count_as_test_failures(
            eager_fail_count=eager_fail_count,
            export_fail_count=export_fail_count,
            fail_count=fail_count,
        )
        if counted_failures and self._should_fail_on_failures():
            self.fail(
                f"{op_name}: {counted_failures} FACTO-generated case(s) failed "
                f"(runtime={fail_count}, export={export_fail_count}, "
                f"eager={eager_fail_count})."
            )


# TODO Figure out where to put these
class FactoTestsXNNPACK(FactoTestsBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(XnnpackTester, *args, **kwargs)

    def _should_skip_case(self, posargs: Sequence[Any]) -> bool:
        if isinstance(posargs[0], torch.Tensor):
            # Temporary for getting around XNN crashes
            # (https://github.com/pytorch/executorch/issues/10960).
            # TODO Re-enable when resolved.
            if posargs[0].dtype in {torch.int8, torch.uint8}:
                print("Skipping (u)int8 case.")
                return True
        return False


for op_name in _selected_op_names():
    FactoTestsXNNPACK._generate_test(op_name)


try:
    from executorch.backends.apple.coreml.test.tester import CoreMLTester

    class FactoTestsCoreML(FactoTestsBase):
        __test__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(CoreMLTester, *args, **kwargs)

    for op_name in _selected_op_names():
        FactoTestsCoreML._generate_test(op_name)

except:
    print("Skipping Core ML facto tests as Core ML AOT is not available.")
