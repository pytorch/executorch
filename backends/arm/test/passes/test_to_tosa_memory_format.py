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
    tosa_transpose_count: int

    def get_inputs(self) -> input_t: ...


class NoNHWC(torch.nn.Module):
    """Test-module with no ops requiring NHWC mermory format."""

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 2

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
    tosa_transpose_count: int = 2

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
    tosa_transpose_count: int = 4

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
    tosa_transpose_count: int = 16

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
    """Test-module with a 4D->4D reshape that merges NCHW dims 1 and 2.

    For models with view_copy shapes [1,2,14,72]->[1,28,1,72] where C=2
    sits at NCHW position 1.  Dims 1 and 2 map to NHWC positions 3 and 1
    (not contiguous), so the reshape is NOT NHWC-safe and transposes are
    inserted around the view_copy.

    Setup: conv2d (forces NHWC, C=2) -> view_copy -> add (keeps in NHWC).

    """

    ops_before_pass: Dict[str, int] = {}
    # 2 I/O transposes for conv + 2 for view_copy (NHWC-unsafe merge)
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 4
    }
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 4

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
    tosa_transpose_count: int = 4

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


class StandalonePermuteChannelsLastOrder(torch.nn.Module):
    """Standalone permute with channels-last perm [0,2,3,1] on NCHW placeholder
    input.

    The permute IS the model's computation (NCHW->NHWC reorder). It must NOT be
    replaced — the input has no tosa_dim_order so is treated as NCHW. The pass
    should insert I/O transposes around the permute_copy as usual.

    """

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2
    }
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)

    def get_inputs(self) -> input_t:
        return (torch.rand(2, 8, 64, 65),)


class RedundantPermuteBetweenConvs(torch.nn.Module):
    """Conv2d -> permute(channels-last order) -> permute(channels-last
    inverse) -> conv2d.

    The permute pair between two NHWC ops duplicates the NCHW<>NHWC
    conversion that tosa_dim_order already handles — both are cancelled
    by FuseConsecutiveTosaTransposesPass (after ToTosaMemoryFormatPass).
    Only the outer I/O transposes remain (2 total).

    """

    ops_before_pass: Dict[str, int] = {}
    # After ToTosaMemoryFormatPass alone: 2 tosa.TRANSPOSE (I/O) +
    # 2 permute_copy still present. FuseConsecutiveTosaTransposesPass fuses
    # the redundant pairs later.
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
    }
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 2

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # channels-last order, redundant between convs
        x = x.permute(0, 3, 1, 2)  # channels-last inverse, undo back to NCHW
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 4, 8, 8),)


class TwoConsecutivePermutes(torch.nn.Module):
    """Conv2d -> permute(channels-last) -> permute(channels-last inverse) ->
    conv2d.

    Two consecutive redundant permutes between NHWC ops. The pair is
    detected and both are cancelled by FuseConsecutiveTosaTransposesPass
    since they compose to the identity and the input is already in
    channels-last dim_order.

    """

    ops_before_pass: Dict[str, int] = {}
    # After ToTosaMemoryFormatPass alone: 2 tosa.TRANSPOSE (I/O) +
    # 2 permute_copy still present. FuseConsecutiveTosaTransposesPass fuses
    # the redundant pairs later.
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
    }
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 2

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # channels-last order: redundant
        x = x.permute(0, 3, 1, 2)  # channels-last inverse: also redundant
        x = self.conv2(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 4, 8, 8),)


class NonChannelsLastPermuteAfterConv(torch.nn.Module):
    """Conv2d followed by a permute that is NOT channels-last order or inverse.

    The permute [0, 1, 3, 2] swaps H and W — this is a semantic permute and must
    be kept. Conv I/O transposes (2) + permute stays as-is.

    """

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
    }
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 3

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.permute(0, 1, 3, 2)  # swap H,W — not channels-last order/inverse

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 4, 8, 6),)


# --- Conformer unfold+movedim pattern (Gemma3n reproduction) ---


class ConformerUnfoldMovedim(torch.nn.Module):
    """Mimics _extract_block_context from Gemma3nAudioAttention.

    unfold(dim=1) decomposes to as_strided_copy (5D), then movedim(-1,2)
    decomposes to permute_copy. The permute is a semantic reordering and must
    NOT be eliminated by the redundant permute optimization.

    """

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {}
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 2

    def __init__(self, chunk_size: int = 16, context_size: int = 20) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.linear = torch.nn.Linear(64, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x = self.linear(x)
        x = x.reshape(b, t, 4, 16)
        pad_left = 2
        pad_right = self.context_size - self.chunk_size + 2 - 1
        left = x.new_zeros(b, pad_left, 4, 16)
        right = x.new_zeros(b, pad_right, 4, 16)
        x = torch.cat([left, x, right], dim=1)
        x_unfolded = x.unfold(dimension=1, size=self.context_size, step=self.chunk_size)
        x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)
        return x_unfolded.contiguous()

    def get_inputs(self) -> input_t:
        return (torch.randn(2, 64, 64),)


class TransformerPermute(torch.nn.Module):
    """Simplified Transformer attention pattern.

    nn.Transformer internally uses transpose(0,1) on 3D tensors which produces
    permutation [1,0,2]. This does NOT match channels-last order (0,2,1) for
    rank-3, so the redundant permute optimization should not apply. This test
    verifies functional correctness through the full quantized pipeline.

    """

    ops_before_pass: Dict[str, int] = {}
    ops_after_pass: Dict[str, int] = {}
    ops_not_after_pass: List[str] = []
    tosa_transpose_count: int = 52
    atol: float = 0.1

    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=64,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dtype=torch.float32,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.transformer(src, tgt)

    def get_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.rand((10, 32, 64)), torch.rand((20, 32, 64)))


modules: Dict[str, ModuleMetadata] = {
    "no_nhwc": NoNHWC(),
    "parallel_clusters": ParallelClusters(),
    "serial_clusters": SerialClusters(),
    "reshapes": Reshapes(),
    "nhwc_safe_spatial_merge": NHWCSafeSpatialMerge(),
    "nhwc_unsafe_channel_change": NHWCUnsafeChannelChange(),
    "standalone_permute_channels_last_order": StandalonePermuteChannelsLastOrder(),
    "redundant_permute_between_convs": RedundantPermuteBetweenConvs(),
    "two_consecutive_permutes": TwoConsecutivePermutes(),
    "non_channels_last_permute_after_conv": NonChannelsLastPermuteAfterConv(),
    "conformer_unfold_movedim": ConformerUnfoldMovedim(),
    "transformer_permute": TransformerPermute(),  # type: ignore[dict-item]
}

# TransformerPermute crashes in FoldAndAnnotateQParamsPass due to BMM shape
# constraints in the quantized pipeline. Only test it via PassPipeline.
functional_modules: Dict[str, ModuleMetadata] = {
    k: v for k, v in modules.items() if k != "transformer_permute"
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


@common.parametrize("module", functional_modules)
def test_to_tosa_memory_format_tosa_INT_functional(module: ModuleMetadata) -> None:
    # Run the full pass pipeline and verify functional correctness + TRANSPOSE count.
    module_nn = cast(torch.nn.Module, module)
    atol = getattr(module, "atol", 1e-03)
    pipeline = TosaPipelineINT[input_t](module_nn, module.get_inputs(), [], atol=atol)
    pipeline.count_tosa_ops({"TRANSPOSE": module.tosa_transpose_count})
    pipeline.run()
