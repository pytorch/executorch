# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_layernorm_pass import (
    DecomposeLayerNormPass,
)

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class LayerNorm(torch.nn.Module):
    """
    Basic layer_norm model using torch.nn.layer_norm layer
    """

    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer_norm(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)


def test_decompose_layernorm_tosa_FP():
    module = LayerNorm()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_native_layer_norm_default": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
            "executorch_exir_dialects_edge__ops_aten_view_copy_default",
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_full_default",
            "executorch_exir_dialects_edge__ops_aten_rsqrt_default",
            "executorch_exir_dialects_edge__ops_aten_var_correction",
            "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
            "executorch_exir_dialects_edge__ops_aten_mean_dim",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 2,
            "executorch_exir_dialects_edge__ops_aten_full_default": 1,
            "executorch_exir_dialects_edge__ops_aten_rsqrt_default": 1,
            "executorch_exir_dialects_edge__ops_aten_var_correction": 1,
            "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_expand_copy_default"
        ],
        pass_list=[DecomposeLayerNormPass],
    )
    pipeline.run()
