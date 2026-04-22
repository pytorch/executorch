# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List, Protocol, Tuple

import torch
from executorch.backends.arm._passes import (
    AnnotateOutputDimOrderPass,
    ToTosaMemoryFormatPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineINT,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass

input_t = Tuple[torch.Tensor]  # Input x


class ModuleMetadata(Protocol):
    ops_before_pass: Dict[str, int]
    ops_after_pass: Dict[str, int]
    ops_not_after_pass: List[str]

    def get_inputs(self) -> input_t: ...


class NoNHWC(torch.nn.Module):
    """Test-module with no ops requiring NHWC mermory format."""

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 2, 2, 2),)


class ParallelClusters(torch.nn.Module):
    """Test-module with multiple parallel clusters of nodes requiring different
    memory formats.
    """

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=1,
            bias=True,
        )
        self.maxpool = torch.nn.MaxPool2d(1, 1)
        self.avgpool = torch.nn.AvgPool2d(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.maxpool(x)
        x3 = self.avgpool(x)
        x4 = x * x
        return x1 + x2 + x3 + x4

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 2, 2, 2),)


class SerialClusters(torch.nn.Module):
    """Test-module with multiple serial clusters of nodes requring different
    memory formats.
    """

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 4
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=1,
            bias=True,
        )
        self.fc = torch.nn.Linear(
            in_features=2,
            out_features=2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x * x
        x = self.conv(x)
        x = x.view((2, 1, 2, 4))
        x = x * 2
        x = x.view((2, 2, 2, 2))
        x = self.conv(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(2, 2, 2, 2),)


class Reshapes(torch.nn.Module):
    """Test-module with different configurations of views requiring different
    memory formats.
    """

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 16
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(1, 1)  # Use maxpool to force NHWC format

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.maxpool(x)
        x = x.view((2, 2, 4, 16, 1))  # N-C-HW-invariant intact, no transposes needed
        x = x * 2  # Add op to avoid views merging
        x = x.view((4, 4, 4, 4))
        x = x / 2  # Add op to avoid views merging
        x = self.maxpool(x)

        x = x.view((256))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((16, 16))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((16, 4, 4))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((2, 4, 4, 8))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((8, 1, 2, 4, 4))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = self.maxpool(x)

        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4, 4, 4),)


class NHWCSafeSpatialMerge(torch.nn.Module):
    """Test-module with a 4D->4D reshape that merges spatial dims H*W while
    preserving the last-dim channel.

    For models with view_copy shapes [1,2,14,72]->[1,28,1,72] where C=2
    sits at NCHW position 1 and the last dim (72) is the NHWC channel that gets
    preserved.  ``_is_nhwc_safe_reshape`` detects that shape_indices on the raw
    shapes are monotonic with the last dim alone, so no transposes are inserted
    around the view_copy.

    Setup: conv2d (forces NHWC, C=2) -> view_copy -> add (keeps in NHWC).

    """

    ops_before_pass: Dict[str, int] = {}
    # Only the 2 I/O transposes for the conv, NO extra transposes from view_copy
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=2, out_channels=2, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # forces NHWC path; output [1, 2, 14, 72]
        x = x.view(1, 28, 1, 72)  # spatial merge: H*W=2*14->28, last dim 72 preserved
        return x + x  # keep result 4-D in NHWC

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 2, 14, 72),)


class NHWCUnsafeChannelChange(torch.nn.Module):
    """Test-module with a 4D->4D reshape that is NOT NHWC-safe because the
    target shape cannot be produced by a monotonic merge of NHWC input dims.

    The pass MUST still insert transposes around the view_copy.

    """

    ops_before_pass: Dict[str, int] = {}
    # conv I/O transposes (2) + view_copy transposes (2) = 4
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 4
    }
    ops_not_after_pass: List[str] = []

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=72, out_channels=72, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # output [1, 72, 2, 14]
        x = x.view(1, 14, 2, 72)  # not NHWC-safe (channels shuffled)
        return x + x

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 72, 2, 14),)


modules: Dict[str, ModuleMetadata] = {
    "no_nhwc": NoNHWC(),
    "parallel_clusters": ParallelClusters(),
    "serial_clusters": SerialClusters(),
    "reshapes": Reshapes(),
    "nhwc_safe_spatial_merge": NHWCSafeSpatialMerge(),
    "nhwc_unsafe_channel_change": NHWCUnsafeChannelChange(),
}


@common.parametrize("module", modules)
def test_to_tosa_memory_format_tosa_INT(module: ModuleMetadata) -> None:
    # We cannot check op counts after a specific pass with the full pipeline
    module_nn = cast(torch.nn.Module, module)
    pipeline = PassPipeline[input_t](
        module_nn,
        module.get_inputs(),
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        pass_list=[RemoveGetItemPass, AnnotateOutputDimOrderPass],
        passes_with_exported_program=[ToTosaMemoryFormatPass],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # Eager execution is not possible after introducing tosa.TRANSPOSE
    pipeline.run()


@common.parametrize("module", modules)
def test_to_tosa_memory_format_tosa_INT_functional(module: ModuleMetadata) -> None:
    # Also run the actual pass pipeline to ensure functional correctness.
    module_nn = cast(torch.nn.Module, module)
    pipeline = TosaPipelineINT[input_t](module_nn, module.get_inputs(), [])
    pipeline.run()


# --- Direct unit tests for NHWC-safe reshape helpers ---


def test_get_shape_indices_spatial_merge():
    """[1,2,14,72] -> [1,28,1,72]: merge H*W, insert size-1 dim, preserve C."""
    indices = ToTosaMemoryFormatPass._get_shape_indices([1, 2, 14, 72], [1, 28, 1, 72])
    assert indices == [[0], [1, 2], [], [3]]


def test_get_shape_indices_identity():
    """Same shape => each dim maps to itself."""
    indices = ToTosaMemoryFormatPass._get_shape_indices([2, 3, 4], [2, 3, 4])
    assert indices == [[0], [1], [2]]


def test_get_shape_indices_full_merge():
    """[2, 3, 4] -> [24]: merge all dims into one."""
    indices = ToTosaMemoryFormatPass._get_shape_indices([2, 3, 4], [24])
    assert indices == [[0, 1, 2]]


def test_get_shape_indices_incompatible():
    """Sizes that don't divide => None."""
    indices = ToTosaMemoryFormatPass._get_shape_indices([2, 3, 5], [6, 4])
    assert indices is None


def test_get_shape_indices_size_one_insert():
    """[6, 4] -> [6, 1, 4]: inserted size-1 dim in the middle."""
    indices = ToTosaMemoryFormatPass._get_shape_indices([6, 4], [6, 1, 4])
    assert indices is not None
    assert indices == [[0], [], [1]]


def test_is_monotonic_true():
    assert ToTosaMemoryFormatPass._is_monotonic([[0], [1, 2], [], [3]])
    assert ToTosaMemoryFormatPass._is_monotonic([[0], [], [1], [2, 3]])
    assert ToTosaMemoryFormatPass._is_monotonic([[], [0, 1, 2]])


def test_is_monotonic_false():
    assert not ToTosaMemoryFormatPass._is_monotonic([[1], [0]])
    assert not ToTosaMemoryFormatPass._is_monotonic([[0, 2], [1]])


def test_is_nhwc_safe_forward():
    """Shapes already NHWC by the time the pass runs.

    [1,2,14,72] -> [1,28,1,72], sr=2 -> NHWC-safe (spatial merge, C=72
    preserved).

    """
    assert ToTosaMemoryFormatPass._is_nhwc_safe_reshape(
        [1, 2, 14, 72], [1, 28, 1, 72], input_sr=2, output_sr=2
    )


def test_is_nhwc_safe_non_4d():
    """Reshapes below rank 4 are never NHWC-safe."""
    assert not ToTosaMemoryFormatPass._is_nhwc_safe_reshape(
        [6, 4], [24], input_sr=0, output_sr=0
    )
