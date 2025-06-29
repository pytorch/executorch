# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import functools
import traceback
from typing import Any, Callable, List, OrderedDict, Sequence, Tuple
import unittest

import torch
from executorch.backends.test.harness.tester import Tester as TesterBase
from executorch.backends.apple.coreml.test.tester import CoreMLTester
from executorch.backends.xnnpack.test.tester.tester import ToEdgeTransformAndLower, Tester as XnnpackTester
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.specs.model import Constraint, ConstraintProducer as cp, Spec
from facto.inputgen.utils.random_manager import random_manager
from facto.inputgen.variable.type import ScalarDtype
from facto.specdb.db import SpecDictDB
from torch._ops import OpOverload

from .facto_specs import ExtraSpecDB

CombinedSpecDB = SpecDictDB | ExtraSpecDB

COMMON_TENSOR_CONSTRAINTS = [
    cp.Rank.Ge(lambda deps: 1),
    cp.Rank.Le(lambda deps: 4),
    cp.Size.Ge(lambda deps, r, d: 1),
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
        fixed_kwargs: dict[str, Any]
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

class ConvModel(OpModel):
    def forward(self, *args, **kwargs):
        weight, bias, stride, padding, dilation, transposed, output_padding, groups = self.fixed_args

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
            
            return op(args[0], weight, bias, stride, padding, output_padding, groups, dilation)

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
        test_body = lambda self: self._test_op(torch_op)

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
                inspec.type.is_tensor() and 
                inspec.name.lower() in RUNTIME_INPUT_NAMES
            )
            if is_runtime_input:
                runtime_input_count += 1
            else:
                break
        
        return max(1, runtime_input_count)

    def setUp(self):
        torch.set_printoptions(threshold=3)
    
    def _test_op(self, op: OpOverload) -> None:
        random_manager.seed(0)

        # Strip namespace
        op_name = op.name().split("::")[-1]

        # Default to .default overload
        if "." not in op_name:
            op_name += ".default"
        
        # Find and patch op spec
        if not op_name in CombinedSpecDB:
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
            #print(f"[{i}]")
            i += 1

            try:
                if isinstance(posargs[0], torch.Tensor):
                    # Temporary for getting around XNN crashes
                    if posargs[0].dtype not in {torch.float32, torch.float16}:
                        print("SKIPPING NON FLOAT CASE")
                        continue
                    if posargs[0].dtype in {torch.int8, torch.uint8}:
                        print("SKIPPING (U)INT8 CASE")
                        continue

                module_cls = get_module_for_op(op)
                model = module_cls(
                    op,
                    runtime_input_count,
                    posargs[runtime_input_count:],
                    inkwargs
                )

                # Sanity check to make sure it runs in eager. This can present nicer error
                # messages sometimes compared to tracing.
                try:
                    model(*posargs[:runtime_input_count])
                except Exception as e:
                    print(f"Eager execution failed: {e}")
                    continue

                tester = (
                    self._tester_factory(
                        model,
                        tuple(posargs[:runtime_input_count])
                    )
                    .export()
                    .dump_artifact()
                    #.to_edge_transform_and_lower(ToEdgeTransformAndLower(partitioners=[]))
                    .to_edge_transform_and_lower()
                    #.dump_artifact()
                )

                is_delegated = any(
                    n.target == torch._higher_order_ops.executorch_call_delegate
                    for n in tester.stages[tester.cur].graph_module.graph.nodes
                    if n.op == "call_function"
                )

                # Only run the runtime test if the op was delegated.
                if is_delegated:
                    (
                        tester
                        .to_executorch()
                        .serialize()
                        .run_method_and_compare_outputs()
                    )
                
                if is_delegated:
                    success_count_delegated += 1
                else:
                    success_count_undelegated += 1
            #finally:
            except Exception as e:
                fail_count += 1
                print(f"Args:")
                for arg in posargs:
                    if isinstance(arg, torch.Tensor):
                        print(f"  {arg.dtype} {arg.shape}")
                    else:
                        print(f"  {arg}")

                traceback.print_exc()

        print(f"{success_count_delegated + success_count_undelegated} PASS, {fail_count} FAIL")
        print(f"  {success_count_delegated} DELEGATED, {success_count_undelegated} UNDELEGATED")

# Programatically generate tests for each operator.
for op_name in CombinedSpecDB.keys():
    FactoTestsBase._generate_test(op_name)

# TODO Figure out where to put these
class FactoTestsXNNPACK(FactoTestsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(XnnpackTester, *args, **kwargs)

class FactoTestsCoreML(FactoTestsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CoreMLTester, *args, **kwargs)
