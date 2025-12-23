# Copyright 2025 Arm Limited and/or its affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Tests 10 popular torch ops, not tested in other ways, training related or requring randomness.
- t
- zeros
- ones
- stack
- arange
- norm
- nonzero
- eye
- topk
- sort
"""

from typing import Callable

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)


def module_add_factory(function: Callable) -> torch.nn.Module:
    class ModuleWrapper(torch.nn.Module):
        def forward(self, x, *args):
            return x + function(*args).to(torch.float32)

    return ModuleWrapper()


def module_factory(function: Callable) -> torch.nn.Module:
    class ModuleWrapper(torch.nn.Module):
        def forward(self, *args):
            return function(*args)

    return ModuleWrapper()


example_input = torch.rand(1, 6, 16, 16)

module_tests = [
    (
        "t",
        module_add_factory(torch.t),
        (
            torch.rand(10, 6),
            torch.rand(6, 10),
        ),
    ),
    (
        "zeros",
        module_add_factory(torch.zeros),
        (
            torch.rand(4, 3, 2),
            (4, 1, 2),
        ),
    ),
    (
        "ones",
        module_add_factory(torch.ones),
        (
            torch.rand(4, 3, 2),
            (4, 1, 2),
        ),
    ),
    (
        "stack",
        module_add_factory(torch.stack),
        (
            torch.rand(1, 1, 1, 1),
            (torch.rand(2, 3, 3), torch.rand(2, 3, 3)),
            -2,
        ),
    ),
    ("arange", module_add_factory(torch.arange), (torch.rand(1), 0, 10, 2)),
    ("norm", module_factory(torch.norm), (torch.randn(5, 5),)),
    ("nonzero", module_factory(torch.nonzero), (example_input,)),
    ("eye", module_add_factory(torch.eye), (torch.rand(4, 4), 4)),
    ("topk", module_factory(torch.topk), (torch.rand(10), 5)),
    ("sort", module_factory(torch.sort), (torch.rand(5),)),
]

input_t = tuple[torch.Tensor]

test_parameters = {test[0]: test[1:] for test in module_tests}


@parametrize(
    "test_data",
    test_parameters,
    xfails={
        "nonzero": "torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u4, 0). "
        "Requires dynamic output shape.",
        "topk": "NotImplementedError: No registered serialization name for <class 'torch.return_types.topk'> found",
        "sort": "NotImplementedError: No registered serialization name for <class 'torch.return_types.sort'> found",
    },
)
def test_torch_functions_tosa_FP(test_data):
    module, inputs = test_data
    pipeline = TosaPipelineFP[input_t](
        module, inputs, "", use_to_edge_transform_and_lower=True
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")
    try:
        pipeline.run()
    except RuntimeError as e:
        if (
            "Ran model with TosaReferenceModelDispatch but never ran TOSABackend delegate."
            not in str(e)
        ):
            raise e


@parametrize(
    "test_data",
    test_parameters,
    xfails={
        "nonzero": "torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u4, 0). "
        "Requires dynamic output shape.",
        "topk": "NotImplementedError: No registered serialization name for <class 'torch.return_types.topk'> found",
        "sort": "NotImplementedError: No registered serialization name for <class 'torch.return_types.sort'> found",
    },
    strict=True,
)
def test_torch_functions_tosa_INT(test_data):
    module, inputs = test_data
    pipeline = TosaPipelineINT[input_t](
        module, inputs, "", use_to_edge_transform_and_lower=True
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.pop_stage("check_not.quant_nodes")

    try:
        pipeline.run()
    except RuntimeError as e:
        if (
            "Ran model with TosaReferenceModelDispatch but never ran TOSABackend delegate."
            not in str(e)
        ):
            raise e
