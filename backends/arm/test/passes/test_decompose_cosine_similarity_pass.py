# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch

from executorch.backends.arm._passes.decompose_cosine_similarity_pass import (
    DecomposeCosineSimilarityPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor, torch.Tensor]


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class CosineSimilarityModel(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(2, 3, 4), torch.rand(2, 3, 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cosine_similarity(x1, x2, dim=1, eps=1e-6)


modules: Dict[str, ModuleWithInputs] = {"cosine_basic": CosineSimilarityModel()}


@common.parametrize("module", modules)
def test_decompose_cosine_similarity_tosa_INT(module: ModuleWithInputs) -> None:

    ops_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 5,
        "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 3,
        "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 2,
        # TODO(masnesral): uncomment after https://github.com/pytorch/pytorch/pull/144765
        # "executorch_exir_dialects_edge__ops_aten_full_default": 1,
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 2,
        "executorch_exir_dialects_edge__ops_aten_reciprocal_default": 1,
    }

    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        ops_before_pass=None,
        ops_not_before_pass=None,
        ops_after_pass=ops_after_pass,
        ops_not_after_pass=None,
        pass_list=[DecomposeCosineSimilarityPass],
        quantize=True,
    )
    pipeline.run()
