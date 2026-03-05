# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FuseConsecutiveTransposesPass.
"""

from typing import Tuple

import torch
from executorch.backends.arm._passes import FuseConsecutiveTransposesPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]


class DoublePermuteIdentityModule(torch.nn.Module):
    """Two permutes that cancel each other out (identity)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC -> NCHW = identity
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 3, 4, 5),)


def test_fuse_consecutive_transposes_identity_removes_both():
    """Test that two canceling permutes are both removed."""
    module = DoublePermuteIdentityModule()
    pipeline = PassPipeline[input_t](
        module,
        DoublePermuteIdentityModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
        },
        ops_after_pass={},
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        ],
        pass_list=[FuseConsecutiveTransposesPass],
    )
    pipeline.run()


class DoublePermuteFusionModule(torch.nn.Module):
    """Two permutes that don't cancel but can be fused into one."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (0,1,2,3) -> (0,2,1,3) -> (0,3,2,1)
        # Combined: (0,1,2,3)[i for i in (0,3,2,1)] applied to (0,2,1,3)
        # = (0,2,1,3) permuted by (0,3,2,1) = [0,3,1,2]
        x = x.permute(0, 2, 1, 3)
        x = x.permute(0, 3, 2, 1)
        return x

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 3, 4, 5),)


def test_fuse_consecutive_transposes_fuses_to_one():
    """Test that two non-canceling permutes are fused into one."""
    module = DoublePermuteFusionModule()
    pipeline = PassPipeline[input_t](
        module,
        DoublePermuteFusionModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[FuseConsecutiveTransposesPass],
    )
    pipeline.run()


class TriplePermuteModule(torch.nn.Module):
    """Three consecutive permutes that should be fused."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (0,1,2,3) -> (0,2,3,1)
        x = x.permute(0, 2, 3, 1)  # -> (0,3,1,2)
        x = x.permute(0, 2, 3, 1)  # -> (0,1,2,3) = identity!
        return x

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 3, 4, 5),)


def test_fuse_consecutive_transposes_triple_identity():
    """Test that three permutes that form identity are all removed."""
    module = TriplePermuteModule()
    pipeline = PassPipeline[input_t](
        module,
        TriplePermuteModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 3,
        },
        ops_after_pass={},
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        ],
        pass_list=[FuseConsecutiveTransposesPass],
    )
    pipeline.run()


class SinglePermuteModule(torch.nn.Module):
    """Single permute that should not be affected."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 3, 4, 5),)


def test_fuse_consecutive_transposes_single_permute_unchanged():
    """Test that a single permute is not affected."""
    module = SinglePermuteModule()
    pipeline = PassPipeline[input_t](
        module,
        SinglePermuteModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[FuseConsecutiveTransposesPass],
    )
    pipeline.run()


class PermuteWithOpBetweenModule(torch.nn.Module):
    """Permutes separated by another op should not be fused."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = x + 1.0  # Op between permutes
        x = x.permute(0, 3, 1, 2)
        return x

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 3, 4, 5),)


def test_fuse_consecutive_transposes_not_consecutive():
    """Test that permutes separated by other ops are not fused."""
    module = PermuteWithOpBetweenModule()
    pipeline = PassPipeline[input_t](
        module,
        PermuteWithOpBetweenModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
        },
        pass_list=[FuseConsecutiveTransposesPass],
    )
    pipeline.run()


class Permute3DIdentityModule(torch.nn.Module):
    """Two 3D permutes that cancel each other out."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L) = identity
        return x

    @staticmethod
    def input() -> input_t:
        return (torch.randn(2, 3, 4),)


def test_fuse_consecutive_transposes_3d_identity():
    """Test that two 3D canceling permutes are removed."""
    module = Permute3DIdentityModule()
    pipeline = PassPipeline[input_t](
        module,
        Permute3DIdentityModule.input(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
        },
        ops_after_pass={},
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        ],
        pass_list=[FuseConsecutiveTransposesPass],
    )
    pipeline.run()
