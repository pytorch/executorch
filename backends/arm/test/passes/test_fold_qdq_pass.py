# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ClassVar, Dict, Tuple

import torch
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline


input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class SimpleQuantizeModel(torch.nn.Module):
    test_data: ClassVar[Dict[str, input_t]] = {
        "rand": (torch.rand(1, 1280, 7, 7), torch.rand(1, 1280, 7, 7)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + torch.max((x + x), (y + y))


@common.parametrize("test_data", SimpleQuantizeModel.test_data)
def test_fold_qdq_pass_tosa_INT(test_data: input_t) -> None:
    """
    Tests the FoldAndAnnotateQParamsPass which folds dq/q nodes into
    the node and stores the quantization parameters in meta.

    Check that the pass runs for add operation and that one q node and one dq node
    is removed from the representation.
    """
    module = SimpleQuantizeModel()
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=True,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 7,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 6,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        },
        pass_list=[FoldAndAnnotateQParamsPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output
    pipeline.run()
