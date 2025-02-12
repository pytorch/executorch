# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm.test.tester.test_pipeline import TestPassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Sigmoid(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return x.sigmoid()

    def get_inputs(self) -> input_t:
        return (torch.rand(4),)


def test_insert_table_tosa_BI():
    module = Sigmoid()
    pipeline = TestPassPipeline[input_t](
        module,
        module.get_inputs(),
        tosa_version="TOSA-0.80+BI",
        ops_before_pass={},
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
            "tosa._table": 1,
        },
        ops_not_after_pass=["aten_sigmoid_default"],
        pass_list=[FoldAndAnnotateQParamsPass],
        passes_with_exported_program=[InsertTableOpsPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output

    pipeline.run()
