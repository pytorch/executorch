# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes.convert_split_to_slice import (
    ConvertSplitToSlicePass,
)

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


class Split(torch.nn.Module):
    """
    Basic split model using torch.split function
    """

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return torch.split(x, 2)


class SplitTensor(torch.nn.Module):
    """
    Basic split model using torch.Tensor.split function
    """

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return x.split(2)


modules: Dict[str, ModuleWithInputs] = {
    "split_basic": Split(),
    "split_tensor": SplitTensor(),
}


@common.parametrize("module", modules)
def test_convert_split_to_slice_tosa_INT(module: ModuleWithInputs) -> None:
    nn_module = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        nn_module,
        module.get_inputs(),
        quantize=True,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 5,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default"
        ],
        pass_list=[ConvertSplitToSlicePass],
    )
    pipeline.run()
