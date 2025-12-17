# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ClassVar, Dict, Tuple

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Sigmoid(torch.nn.Module):
    test_data: ClassVar[Dict[str, input_t]] = {
        "rand": (torch.rand(4),),
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid()


@common.parametrize("test_data", Sigmoid.test_data)
def test_insert_table_tosa_INT(test_data: input_t) -> None:
    module = Sigmoid()
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=True,
        ops_before_pass={"executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1},
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
            "backend__ops_tosa_TABLE_default": 1,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_aten_sigmoid_default"],
        pass_list=[FoldAndAnnotateQParamsPass],
        passes_with_exported_program=[InsertTableOpsPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output

    pipeline.run()
