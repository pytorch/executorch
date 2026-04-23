# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.remove_getitem_pass import RemoveGetItemPass
from executorch.backends.arm._passes.rewrite_max_pool2d_pass import RewriteMaxPool2dPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class MaxPool2dWithStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)


class MaxPool2dWithoutStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=3)


class MaxPool2dListKernel(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, kernel_size=[2, 3])


modules: Dict[str, ModuleWithInputs] = {
    "max_pool2d_with_stride": MaxPool2dWithStride(),
    "max_pool2d_without_stride": MaxPool2dWithoutStride(),
    "max_pool2d_list_kernel": MaxPool2dListKernel(),
}


@common.parametrize("module", modules)
def test_rewrite_max_pool2d_tosa(module: ModuleWithInputs) -> None:
    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_MAX_POOL2D_default": 1,
        },
        pass_list=[RemoveGetItemPass, RewriteMaxPool2dPass],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # Cannnot run aten graph with tosa dialect ops
    pipeline.run()
