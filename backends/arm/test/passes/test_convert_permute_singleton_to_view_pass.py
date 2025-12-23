# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm._passes import ConvertPermuteSingletonToViewPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]


class PermuteSingletonAxesModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 1, 3, 4),)


def test_convert_permute_singleton_to_view_tosa_FP_applies():
    module = PermuteSingletonAxesModule()
    pipeline = PassPipeline[input_t](
        module,
        PermuteSingletonAxesModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        ],
        pass_list=[ConvertPermuteSingletonToViewPass],
    )
    pipeline.run()


class PermuteNonSingletonModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 3, 4),)


def test_convert_permute_singleton_to_view_tosa_FP_skip_non_singleton():
    module = PermuteNonSingletonModule()
    pipeline = PassPipeline[input_t](
        module,
        PermuteNonSingletonModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_view_copy_default",
        ],
        pass_list=[ConvertPermuteSingletonToViewPass],
    )
    pipeline.run()


class PermuteSameSizedNonSingletonModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(2, 1, 0)

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 1, 2),)


def test_convert_permute_singleton_to_view_tosa_FP_skip_same_size_non_singleton():
    module = PermuteSameSizedNonSingletonModule()
    pipeline = PassPipeline[input_t](
        module,
        PermuteSameSizedNonSingletonModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_view_copy_default",
        ],
        pass_list=[ConvertPermuteSingletonToViewPass],
    )
    pipeline.run()
