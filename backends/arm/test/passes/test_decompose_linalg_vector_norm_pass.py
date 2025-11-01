# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch

from executorch.backends.arm._passes.decompose_linalg_vector_norm_pass import (
    DecomposeLinearVectorNormPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]


class ModuleWithInputs(Protocol):
    ord: float | None

    def get_inputs(self) -> input_t: ...


class VectorNormModel(torch.nn.Module):
    """
    A test module with torch.linalg.vector_norm.
    https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html

    We support only order 1 or 2.
    """

    def __init__(
        self, ord: float | None = None, dim=None, keepdim: bool = False
    ) -> None:
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ord is None and self.dim is None:
            return torch.linalg.vector_norm(x, keepdim=self.keepdim)
        elif self.ord is None:
            return torch.linalg.vector_norm(x, dim=self.dim, keepdim=self.keepdim)
        elif self.dim is None:
            return torch.linalg.vector_norm(x, ord=self.ord, keepdim=self.keepdim)
        else:
            return torch.linalg.vector_norm(
                x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
            )

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


modules = {
    # Default uses p=2 (l2 vector norm)
    "default_p2": VectorNormModel(dim=1),
    # p = 1: L1 norm over all elements
    "p1": VectorNormModel(ord=1, dim=1),
}


@common.parametrize("module", modules)
def test_decompose_vector_norm_tosa_INT(module: ModuleWithInputs) -> None:
    """
    This test creates a PassPipeline that applies the DecomposeLinearVectorNormPass.
    The expected primitive ops vary depending on the norm order:
      - p == 1: should decompose to ABS and SUM.
      - p == 2 (default): should decompose to MUL, SUM, and SQRT.
      - Other p: should decompose to ABS, two instances of POW, and SUM.
    """
    ord_val = module.ord if module.ord is not None else 2.0

    ops_after_pass: Dict[str, int]
    if ord_val == 1:
        ops_after_pass = {
            "executorch_exir_dialects_edge__ops_aten_abs_default": 1,
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 1,
        }
    elif ord_val == 2:
        ops_after_pass = {
            "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 2,
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 1,
        }
    else:
        ops_after_pass = {
            "executorch_exir_dialects_edge__ops_aten_abs_default": 1,
            "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 2,
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 1,
        }

    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        # The op is decomposed in legalization aten -> edge, so we are not able to check ops before
        ops_before_pass=None,
        ops_not_before_pass=None,
        ops_after_pass=ops_after_pass,
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_linarg_vector_norm_default",
        ],
        pass_list=[DecomposeLinearVectorNormPass],
    )
    pipeline.run()
