# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
# python -m unittest backends.test.operators.test_facto.FactoTestsXNNPACK
#

import copy
import functools
import traceback
import unittest
from typing import Any, Callable, Sequence

import torch
from executorch.backends.test.harness.tester import Tester as TesterBase
from executorch.backends.xnnpack.test.tester.tester import Tester as XnnpackTester
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.specs.model import ConstraintProducer as cp, Spec
from facto.inputgen.utils.random_manager import random_manager
from facto.specdb.db import SpecDictDB
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
    def __init__(self, tester_factory: Callable[[], TesterBase], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tester_factory = tester_factory

    @staticmethod
    def _generate_test(op_name: str) -> None:
        # Find the torch op with the given name.
        sections = op_name.split(".")
        torch_op = functools.reduce(getattr, sections, torch.ops.aten)

        test_name = "test_" + op_name.replace(".", "_")

        def test_body(self):
            self._test_op(torch_op)

        setattr(FactoTestsBase, test_name, test_body)

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

        runtime_input_count = FactoTestsBase.get_runtime_input_count(spec)

        print(f"Op: {op_name}, {runtime_input_count} runtime inputs")

        # Run test cases
        success_count_delegated = 0
        success_count_undelegated = 0
        fail_count = 0

        i = 0
        for posargs, inkwargs, _ in ArgumentTupleGenerator(spec).gen():
            i += 1

            try:
                if isinstance(posargs[0], torch.Tensor):
                    # Temporary for getting around XNN crashes (https://github.com/pytorch/executorch/issues/10960).
                    # TODO Re-enable when resolved.
                    if posargs[0].dtype in {torch.int8, torch.uint8}:
                        print("Skipping (u)int8 case.")
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
                    print(f"Eager execution failed: {e}")
                    continue

                tester = self._tester_factory(
                    model, tuple(posargs[:runtime_input_count])
                )

                # Dynamo will also fail to handle some patterns that are valid in eager.
                try:
                    tester.export()
                except Exception:
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
                    (
                        tester.to_executorch()
                        .serialize()
                        .run_method_and_compare_outputs()
                    )

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


# Programatically generate tests for each operator.
for op_name in CombinedSpecDB.keys():
    FactoTestsBase._generate_test(op_name)


# TODO Figure out where to put these
class FactoTestsXNNPACK(FactoTestsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(XnnpackTester, *args, **kwargs)


try:
    from executorch.backends.apple.coreml.test.tester import CoreMLTester

    class FactoTestsCoreML(FactoTestsBase):
        def __init__(self, *args, **kwargs):
            super().__init__(CoreMLTester, *args, **kwargs)

except:
    print("Skipping Core ML facto tests as Core ML AOT is not available.")
