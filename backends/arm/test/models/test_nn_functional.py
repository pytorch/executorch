# Copyright 2025 Arm Limited and/or its affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests 10 popular torch.nn.functional not tested in other ways or training related
- normalize
- grid_sample
- one_hot
- softplus
- cosine_similarity
- unfold
- elu
- fold
- affine_grid
- max_pool1d
- threshold
"""
from typing import Callable

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)


def module_factory(function: Callable) -> torch.nn.Module:
    class ModuleWrapper(torch.nn.Module):
        def forward(self, *args):
            return function(*args)

    return ModuleWrapper()


example_input = torch.rand(1, 6, 16, 16)

module_tests = {
    "normalize": (module_factory(torch.nn.functional.normalize), (example_input,)),
    "grid_sample": (
        module_factory(torch.nn.functional.grid_sample),
        (torch.rand(1, 1, 4, 4), torch.rand(1, 5, 5, 2)),
    ),
    "one_hot": (
        module_factory(torch.nn.functional.one_hot),
        (torch.randint(0, 5, (2, 2, 5, 5)), 5),
    ),
    "softplus": (module_factory(torch.nn.functional.softplus), (example_input,)),
    "cosine_similarity": (
        module_factory(torch.nn.functional.cosine_similarity),
        (example_input, example_input),
    ),
    "unfold": (
        module_factory(torch.nn.functional.unfold),
        (torch.randn(1, 3, 10, 12), (4, 5)),
    ),
    "elu": (module_factory(torch.nn.functional.elu), (example_input,)),
    "fold": (
        module_factory(torch.nn.functional.fold),
        (torch.randn(1, 12, 12), (4, 5), (2, 2)),
    ),
    "affine_grid": (
        module_factory(torch.nn.functional.affine_grid),
        (torch.rand(1, 2, 3), (1, 2, 10, 10)),
    ),
    "max_pool1d": (
        module_factory(torch.nn.functional.max_pool1d),
        (torch.randn(20, 16, 50), 4),
    ),
    "threshold": (
        module_factory(torch.nn.functional.threshold),
        (example_input, 0.5, 0.1),
    ),
}

input_t = tuple[torch.Tensor]


@parametrize(
    "test_data",
    module_tests,
)
def test_nn_functional_FP(test_data):
    module, inputs = test_data
    pipeline = TosaPipelineFP[input_t](
        module, inputs, "", use_to_edge_transform_and_lower=False
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
    module_tests,
    {"normalize": "MLETORCH-1255: Unsupported dtype in InsertTableOpsPass"},
)
def test_nn_functional_INT(test_data):
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
