# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import DecomposeQuantNodesPass
from executorch.backends.arm.test.common import parametrize
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline


class Mul(torch.nn.Module):
    test_data = {
        "randn": (torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)),
        "large_randn": (10e10 * torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)),
    }

    def forward(self, x, y):
        return x * y


@parametrize("test_data", Mul.test_data)
def test_decompose_quant_nodes_pass(test_data: Tuple[torch.Tensor]):
    module = Mul()
    q_dq_ops = {
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }
    # Verify that DecomposeQuantNodesPass removes quantize/dequantize nodes
    # and that the output is correct.
    pipeline = PassPipeline(
        module,
        test_data,
        quantize=True,
        pass_list=[
            DecomposeQuantNodesPass,
        ],
        ops_before_pass=q_dq_ops,
        ops_not_after_pass=list(q_dq_ops.keys()),
        tosa_extensions=["FP"],
    )
    pipeline.run()
