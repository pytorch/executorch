# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.convert_split_to_slice import (
    ConvertSplitToSlicePass,
)

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class Split(torch.nn.Module):
    """
    Basic split model using torch.split function
    """

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)

    def forward(self, x):
        return torch.split(x, 2)


class SplitTensor(torch.nn.Module):
    """
    Basic split model using torch.Tensor.split function
    """

    def get_inputs(self) -> input_t:
        return (torch.rand(10),)

    def forward(self, x):
        return x.split(2)


modules = {"split_basic": Split(), "split_tensor": SplitTensor()}


@common.parametrize("module", modules)
def test_split_to_slice_tosa_INT(module):
    pipeline = PassPipeline[input_t](
        module,
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
