# Copyright 2025 Arm Limited and/or its affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests 10 popular nn modules not tested in other ways or training related.
- Embedding
- LeakyReLU
- BatchNorm1d
- AdaptiveAvgPool2d
- ConvTranspose2d
- GRU
- GroupNorm
- InstanceNorm2d
- PReLU
- Transformer
"""

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)

example_input = torch.rand(1, 6, 16, 16)

module_tests = [
    (torch.nn.Embedding(10, 10), (torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]),)),
    (torch.nn.LeakyReLU(), (example_input,)),
    (torch.nn.BatchNorm1d(16), (torch.rand(6, 16, 16),)),
    (torch.nn.AdaptiveAvgPool2d((12, 12)), (example_input,)),
    (torch.nn.ConvTranspose2d(6, 3, 2), (example_input,)),
    (torch.nn.GRU(10, 20, 2), (torch.randn(5, 3, 10), torch.randn(2, 3, 20))),
    (torch.nn.GroupNorm(2, 6), (example_input,)),
    (torch.nn.InstanceNorm2d(16), (example_input,)),
    (torch.nn.PReLU(), (example_input,)),
    (
        torch.nn.Transformer(
            d_model=64,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dtype=torch.float32,
        ),
        (torch.rand((10, 32, 64)), torch.rand((20, 32, 64))),
    ),
]

input_t = tuple[torch.Tensor]

test_parameters = {str(test[0].__class__.__name__): test for test in module_tests}


@parametrize(
    "test_data",
    test_parameters,
)
def test_nn_Modules_FP(test_data):
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
        "GRU": "RuntimeError: Node aten_linear_default with op <EdgeOpOverload: aten.linear[...]> was not decomposed or delegated.",
        "PReLU": "RuntimeError: mul(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.",
        "Transformer": "AssertionError: Output 0 does not match reference output.",
    },
)
def test_nn_Modules_INT(test_data):
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
