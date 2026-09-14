# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import ClassVar, Dict, Tuple

import torch
from executorch.backends.arm._passes.convert_to_clamp_pass import ConvertToClampPass
from executorch.backends.arm._passes.fuse_consecutive_clamps_pass import (
    FuseConsecutiveClampsPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor]  # Input x

clamp_op = "executorch_exir_dialects_edge__ops_aten_clamp_default"
hardtanh_op = "executorch_exir_dialects_edge__ops_aten_hardtanh_default"
relu_op = "executorch_exir_dialects_edge__ops_aten_relu_default"


class HardTanhReLU(torch.nn.Module):
    """HardTanh(-1, 1) -> ReLU fuses to clamp(0, 1) (the D110114877 case)."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def __init__(self) -> None:
        super().__init__()
        self.act = torch.nn.Sequential(torch.nn.Hardtanh(-1.0, 1.0), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class ReLUReLU(torch.nn.Module):
    """ReLU -> ReLU fuses to a single clamp(0, None)."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def __init__(self) -> None:
        super().__init__()
        self.act = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class HardTanhHardTanh(torch.nn.Module):
    """HardTanh(-2, 2) -> HardTanh(-1, 3) fuses to clamp(-1, 2)."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def __init__(self) -> None:
        super().__init__()
        self.act = torch.nn.Sequential(
            torch.nn.Hardtanh(-2.0, 2.0), torch.nn.Hardtanh(-1.0, 3.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class ReLU6ReLU(torch.nn.Module):
    """ReLU6 (== HardTanh(0, 6)) -> ReLU fuses to clamp(0, 6)."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def __init__(self) -> None:
        super().__init__()
        self.act = torch.nn.Sequential(torch.nn.ReLU6(), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class ClampClamp(torch.nn.Module):
    """Explicit clamp(-2, 2) -> clamp(-1, 3) fuses to clamp(-1, 2)."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -2.0, 2.0)
        x = torch.clamp(x, -1.0, 3.0)
        return x


class ConflictingClamp(torch.nn.Module):
    """Empty composed range (min>max) leaves both clamps unfused."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 5.0, 10.0)
        x = torch.clamp(x, 0.0, 3.0)
        return x


class ThreeChain(torch.nn.Module):
    """HardTanh -> ReLU -> ReLU collapses to a single clamp."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def __init__(self) -> None:
        super().__init__()
        self.act = torch.nn.Sequential(
            torch.nn.Hardtanh(-1.0, 1.0), torch.nn.ReLU(), torch.nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class BranchingClamp(torch.nn.Module):
    """First clamp feeds two users, so the pair must NOT be fused."""

    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.randn(1, 8, 8, 3),)}

    def __init__(self) -> None:
        super().__init__()
        self.hardtanh = torch.nn.Hardtanh(-1.0, 1.0)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hardtanh(x)
        r = self.relu(h)
        return r + h


"""
Tests FuseConsecutiveClampsPass, which fuses chains of adjacent clamp.default ops
(produced from hardtanh/relu/relu6/explicit clamp by ConvertToClampPass) into a
single clamp with composed bounds.
"""

# Each fuseable module collapses its clamp chain to exactly one clamp.
_fuse_pass_list: list = [ConvertToClampPass, FuseConsecutiveClampsPass]


@common.parametrize("test_data", HardTanhReLU.test_data)
def test_fuse_clamps_tosa_FP_hardtanh_relu(test_data: input_t) -> None:
    pipeline = PassPipeline[input_t](
        HardTanhReLU(),
        test_data,
        quantize=False,
        ops_before_pass={hardtanh_op: 1, relu_op: 1},
        ops_after_pass={clamp_op: 1},
        ops_not_after_pass=[hardtanh_op, relu_op],
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", HardTanhReLU.test_data)
def test_hardtanh_relu_without_fusion_leaves_two_clamps(test_data: input_t) -> None:
    # Baseline / parent behavior: ConvertToClampPass alone turns HardTanh -> ReLU
    # into two adjacent clamps -- the "multiple clamps in a row" that
    # FuseConsecutiveClampsPass (test above) collapses to a single clamp.
    pipeline = PassPipeline[input_t](
        HardTanhReLU(),
        test_data,
        quantize=False,
        ops_after_pass={clamp_op: 2},
        pass_list=[ConvertToClampPass],  # type: ignore[arg-type]
    )
    pipeline.run()


@common.parametrize("test_data", ReLUReLU.test_data)
def test_fuse_clamps_tosa_FP_relu_relu(test_data: input_t) -> None:
    pipeline = PassPipeline[input_t](
        ReLUReLU(),
        test_data,
        quantize=False,
        ops_before_pass={relu_op: 2},
        ops_after_pass={clamp_op: 1},
        ops_not_after_pass=[relu_op],
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", HardTanhHardTanh.test_data)
def test_fuse_clamps_tosa_FP_hardtanh_hardtanh(test_data: input_t) -> None:
    pipeline = PassPipeline[input_t](
        HardTanhHardTanh(),
        test_data,
        quantize=False,
        ops_before_pass={hardtanh_op: 2},
        ops_after_pass={clamp_op: 1},
        ops_not_after_pass=[hardtanh_op],
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", ReLU6ReLU.test_data)
def test_fuse_clamps_tosa_FP_relu6_relu(test_data: input_t) -> None:
    pipeline = PassPipeline[input_t](
        ReLU6ReLU(),
        test_data,
        quantize=False,
        ops_after_pass={clamp_op: 1},
        ops_not_after_pass=[relu_op],
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", ClampClamp.test_data)
def test_fuse_clamps_tosa_FP_clamp_clamp(test_data: input_t) -> None:
    pipeline = PassPipeline[input_t](
        ClampClamp(),
        test_data,
        quantize=False,
        ops_before_pass={clamp_op: 2},
        ops_after_pass={clamp_op: 1},
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", ConflictingClamp.test_data)
def test_fuse_clamps_tosa_FP_conflicting_not_fused(test_data: input_t) -> None:
    # Composed range is empty (max(5, 0)=5 > min(10, 3)=3), so the guard leaves
    # both clamps in place rather than emitting an invalid fused clamp.
    pipeline = PassPipeline[input_t](
        ConflictingClamp(),
        test_data,
        quantize=False,
        ops_before_pass={clamp_op: 2},
        ops_after_pass={clamp_op: 2},
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", ThreeChain.test_data)
def test_fuse_clamps_tosa_FP_three_chain(test_data: input_t) -> None:
    pipeline = PassPipeline[input_t](
        ThreeChain(),
        test_data,
        quantize=False,
        ops_before_pass={hardtanh_op: 1, relu_op: 2},
        ops_after_pass={clamp_op: 1},
        ops_not_after_pass=[hardtanh_op, relu_op],
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", BranchingClamp.test_data)
def test_fuse_clamps_tosa_FP_branching_not_fused(test_data: input_t) -> None:
    # First clamp has two users -> both clamps survive (no fusion).
    pipeline = PassPipeline[input_t](
        BranchingClamp(),
        test_data,
        quantize=False,
        ops_before_pass={hardtanh_op: 1, relu_op: 1},
        ops_after_pass={clamp_op: 2},
        pass_list=_fuse_pass_list,
    )
    pipeline.run()


@common.parametrize("test_data", HardTanhReLU.test_data)
def test_fuse_clamps_tosa_INT_hardtanh_relu(test_data: input_t) -> None:
    # End-to-end INT lowering: the fusion runs after q/dq folding and the
    # quantized output still matches (qtol=1) despite dropping the intermediate
    # requantization.
    pipeline = TosaPipelineINT[input_t](
        HardTanhReLU(),
        test_data,
        aten_op=[],
        exir_op=[],
        qtol=1,
    )
    pipeline.run()
