# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import EthosU55PipelineINT
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs


input_t = Tuple[torch.Tensor]


class SimpleModel(torch.nn.Module):
    test_data = {
        "rand_rand": (torch.rand(1, 2, 2, 1), torch.rand(1, 2, 2, 1)),
    }

    def forward(self, x, y):
        return x + y


@common.parametrize("test_data", SimpleModel.test_data)
def test_ioquantisation_pass_u55_INT(test_data: input_t):
    """
    Test the executorch/exir/passes/quanize_io_pass pass works(meaning we don't get Q/DQ nodes) on a simple model
    """
    model = SimpleModel()
    pipeline = EthosU55PipelineINT(
        model,
        test_data,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=False,
        run_on_fvp=False,
    )
    pipeline.pop_stage(-1)
    pipeline.run()
    edge = pipeline.tester.get_artifact()
    edge.transform(passes=[QuantizeInputs(edge, [0, 1]), QuantizeOutputs(edge, [0])])
    pipeline.tester.check_not(["edge__ops_quantized_decomposed_quantize_per_tensor"])
    pipeline.tester.check_not(["edge__ops_quantized_decomposed_dequantize_per_tensor"])
