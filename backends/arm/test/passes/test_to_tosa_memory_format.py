# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List, Protocol, Tuple

import torch
from executorch.backends.arm._passes import (
    AnnotateOutputDimOrderPass,
    EnsureUniqueOutputNodesPass,
    FuseEqualPlaceholdersPass,
    ToTosaMemoryFormatPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.exir.dialects._ops import ops as exir_ops

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


class DuplicateConstantOutputs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("grid0", torch.zeros(1, 32, 32, 2))
        self.register_buffer("grid1", torch.zeros(1, 32, 32, 2))

    def forward(self, x: torch.Tensor):
        return self.grid0, self.grid1, x


class DuplicateConstantOutputsWithAdd(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("grid0", torch.zeros(1, 32, 32, 2))
        self.register_buffer("grid1", torch.zeros(1, 32, 32, 2))

    def forward(self, x: torch.Tensor):
        return self.grid0, self.grid1, x + x


modules: Dict[str, ModuleMetadata] = {
    "no_nhwc": NoNHWC(),
    "parallel_clusters": ParallelClusters(),
    "serial_clusters": SerialClusters(),
    "reshapes": Reshapes(),
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


def test_to_tosa_memory_format_no_target_preserves_duplicate_output_slots() -> None:
    pipeline = PassPipeline[input_t](
        DuplicateConstantOutputs(),
        (torch.rand(1, 2, 32, 32),),
        quantize=False,
        pass_list=[RemoveGetItemPass, AnnotateOutputDimOrderPass],
        passes_with_exported_program=[
            FuseEqualPlaceholdersPass,
            ToTosaMemoryFormatPass,
            EnsureUniqueOutputNodesPass,
        ],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    graph_module = pipeline.tester.get_artifact().exported_program().graph_module
    output_node = graph_module.graph.output_node()
    outputs = list(output_node.args[0])

    assert outputs[0] is not outputs[1]
    assert outputs[0].target == exir_ops.backend.tosa.IDENTITY.default
    assert outputs[1].target == exir_ops.backend.tosa.IDENTITY.default
    assert outputs[0].args[0] is outputs[1].args[0]


def test_to_tosa_memory_format_tosa_FP_duplicate_output_identity() -> None:
    pipeline = TosaPipelineFP[input_t](
        DuplicateConstantOutputsWithAdd(),
        (torch.rand(1, 2, 32, 32),),
        [],
        [],
    )
    pipeline.run()
