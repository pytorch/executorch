# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import DecomposeEmbeddingPass

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Int32Embedding(torch.nn.Module):

    def forward(self, weights: torch.Tensor, indices: torch.Tensor):
        return torch.embedding(weights, indices)

    def get_inputs(self) -> input_t:
        return (
            torch.randn(9, 3),
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        )


def test_int64_model_tosa_FP():
    module = Int32Embedding()
    op_checks_before = {
        "executorch_exir_dialects_edge__ops_aten_embedding_default": 1,
    }
    op_checks_after = {
        "executorch_exir_dialects_edge__ops_aten_view_copy": 2,
        "executorch_exir_dialects_edge__ops_aten_index_select": 1,
    }

    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass=op_checks_before,
        ops_after_pass=op_checks_after,
        pass_list=[DecomposeEmbeddingPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output
    pipeline.run()
