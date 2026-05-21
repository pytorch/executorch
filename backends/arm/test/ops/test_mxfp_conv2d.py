# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from collections.abc import Callable
from typing import Tuple

import torch
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.test import common
from executorch.backends.arm.test.ops.mxfp.common import (
    MXFPTosaPipelineFP,
    MXFPVgfPipeline,
)
from executorch.backends.arm.test.tester.analyze_output_utils import (
    compare_rel_frobenius_and_cosine_similarity,
)

aten_op = "torch.ops.tosa_mxfp.conv2d.default"

input_t1 = Tuple[torch.Tensor]


def _block_input_rank4(
    batches: int = 1,
    height: int = 8,
    width: int = 8,
) -> torch.Tensor:
    """Create a rank-4 input with distinct MXFP activation block scales."""

    first = torch.cat(
        (
            1e-3 * torch.randn(32, height, width),
            100.0 * torch.randn(32, height, width),
        )
    )
    if batches == 1:
        return first.unsqueeze(0)

    second = torch.cat(
        (
            100.0 * torch.randn(32, height, width),
            1e-3 * torch.randn(32, height, width),
        )
    )
    return torch.stack((first, second))


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        input_data: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.input_data = input_data
        self.conv = torch.nn.Conv2d(
            input_shape[1],
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def get_inputs(self) -> torch.Tensor:
        if self.input_data is not None:
            return self.input_data

        return torch.linspace(
            -0.03,
            0.03,
            math.prod(self.input_shape),
        ).reshape(self.input_shape)

    def set_block_test_weights(self) -> None:
        in_channels = self.conv.weight.shape[1]
        if in_channels % 32 != 0:
            raise ValueError(
                f"MXFP Conv2d test in_channels must be divisible by 32, got "
                f"{in_channels}"
            )

        with torch.no_grad():
            self.conv.weight.zero_()
            for block_start in range(0, in_channels, 32):
                block_end = block_start + 32
                if (block_start // 32) % 2 == 0:
                    self.conv.weight[:, block_start:block_end, :, :] = 1e-3
                else:
                    self.conv.weight[:, block_start:block_end, :, :] = 100.0
            if self.conv.bias is not None:
                self.conv.bias.zero_()


def _is_conv2d(module: torch.nn.Module, _fqn: str) -> bool:
    return isinstance(module, torch.nn.Conv2d)


test_data_fp = {
    "mxfp_conv2d_1x1_1x32x8x8_nobias": lambda: (
        Conv2d(
            input_shape=(1, 32, 8, 8),
            out_channels=4,
            kernel_size=1,
            bias=False,
        ),
        False,
    ),
    "mxfp_conv2d_3x3_1x64x8x8_pad1": lambda: (
        Conv2d(
            input_shape=(1, 64, 8, 8),
            out_channels=8,
            kernel_size=3,
            padding=1,
        ),
        False,
    ),
    "mxfp_conv2d_3x3_2x64x9x7_st2_pad1": lambda: (
        Conv2d(
            input_shape=(2, 64, 9, 7),
            out_channels=5,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        False,
    ),
    "mxfp_conv2d_2x3_1x96x10x11_st1x2_pad0x1_nobias": lambda: (
        Conv2d(
            input_shape=(1, 96, 10, 11),
            out_channels=6,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(0, 1),
            bias=False,
        ),
        False,
    ),
    "mxfp_conv2d_3x3_1x64x10x10_dil2": lambda: (
        Conv2d(
            input_shape=(1, 64, 10, 10),
            out_channels=7,
            kernel_size=3,
            padding=2,
            dilation=2,
        ),
        False,
    ),
    "mxfp_conv2d_1x1_block_weights": lambda: (
        Conv2d(
            input_shape=(1, 64, 8, 8),
            out_channels=4,
            kernel_size=1,
            bias=False,
        ),
        True,
    ),
    "mxfp_conv2d_1x1_block_weights_block_activations": lambda: (
        Conv2d(
            input_shape=(1, 64, 8, 8),
            out_channels=4,
            kernel_size=1,
            bias=False,
            input_data=_block_input_rank4(),
        ),
        True,
    ),
    "mxfp_conv2d_3x3_block_weights_block_activations": lambda: (
        Conv2d(
            input_shape=(2, 64, 8, 8),
            out_channels=4,
            kernel_size=3,
            padding=1,
            input_data=_block_input_rank4(batches=2),
        ),
        True,
    ),
}

test_data_vgf_fp = test_data_fp

# TODO: MLETORCH-2141
_vgf_xfail_reason = (
    "MXFP is not yet supported in the VGF toolchain. Enable this test when "
    "toolchain support is available."
)
_vgf_xfails: dict[str, str | tuple[str, type[Exception]]] = {
    test_case: _vgf_xfail_reason for test_case in test_data_vgf_fp
}


def _assert_weight_block_scales_differ(model: Conv2d) -> None:
    weight_scale = model.conv.weight_scale.to(torch.float32)
    assert weight_scale.shape[-1] > 1
    assert torch.unique(weight_scale).numel() > 1


def _test_mxfp_conv2d_tosa_FP(
    test_data,
    config: MXFPOpConfig,
    frobenius_threshold=0.08,
    cosine_threshold=0.995,
) -> None:
    module, set_block_weights = test_data()
    module = module.eval()

    if set_block_weights:
        module.set_block_test_weights()

    test_input = module.get_inputs()

    pipeline = MXFPTosaPipelineFP[input_t1](
        module,
        (test_input,),
        aten_op,
        filter_fn=_is_conv2d,
        frobenius_threshold=frobenius_threshold,
        cosine_threshold=cosine_threshold,
        mxfp_config=config,
        tosa_version="1.1",
        tosa_extensions=["mxfp"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp)
def test_mxfp8_conv2d_tosa_FP(
    test_data: Callable[[], tuple[Conv2d, bool]],
) -> None:
    _test_mxfp_conv2d_tosa_FP(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp4_conv2d_tosa_FP(
    test_data: Callable[[], tuple[Conv2d, bool]],
) -> None:
    _test_mxfp_conv2d_tosa_FP(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
        frobenius_threshold=0.3,
        cosine_threshold=0.95,
    )


def _test_mxfp_conv2d_vgf(
    test_data,
    config: MXFPOpConfig,
    frobenius_threshold,
    cosine_threshold,
) -> None:
    module, set_block_weights = test_data()
    module = module.eval()

    if set_block_weights:
        module.set_block_test_weights()

    test_input = module.get_inputs()

    pipeline = MXFPVgfPipeline[input_t1](
        module,
        (test_input,),
        aten_op,
        filter_fn=_is_conv2d,
        frobenius_threshold=frobenius_threshold,
        cosine_threshold=cosine_threshold,
        mxfp_config=config,
        tosa_spec="TOSA-1.1+FP+mxfp",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_vgf_fp, xfails=_vgf_xfails)
@common.SkipIfNoModelConverter
def test_mxfp8_conv2d_vgf(test_data: Callable[[], tuple[Conv2d, bool]]) -> None:
    _test_mxfp_conv2d_vgf(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
        frobenius_threshold=0.08,
        cosine_threshold=0.995,
    )


@common.parametrize("test_data", test_data_vgf_fp, xfails=_vgf_xfails)
@common.SkipIfNoModelConverter
def test_mxfp4_conv2d_vgf(test_data: Callable[[], tuple[Conv2d, bool]]) -> None:
    _test_mxfp_conv2d_vgf(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
        frobenius_threshold=0.3,
        cosine_threshold=0.95,
    )


def _test_mxfp_conv2d_eager_cpu(
    test_data,
    config: MXFPOpConfig,
    frobenius_threshold=0.08,
    cosine_threshold=0.995,
) -> None:
    ref_model, set_block_weights = test_data()
    ref_model = ref_model.eval()
    if set_block_weights:
        ref_model.set_block_test_weights()
    test_model = copy.deepcopy(ref_model).eval()
    test_input = ref_model.get_inputs()

    to_mxfp(test_model, config, filter_fn=_is_conv2d)
    if set_block_weights:
        _assert_weight_block_scales_differ(test_model)

    test_output = test_model(test_input)
    ref_output = ref_model(test_input)

    compare_rel_frobenius_and_cosine_similarity(
        ref_output,
        test_output,
        quantization_parameters=None,
        frobenius_threshold=frobenius_threshold,
        cosine_threshold=cosine_threshold,
        clean_reference=False,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp8_conv2d_eager_cpu(
    test_data: Callable[[], tuple[Conv2d, bool]],
) -> None:
    """Check eager MXFP implementation.

    The Arm lowering tests compare lowered output against the eager CPU
    implementation, so the eager implementation must be accurate for it to be
    used as a reference in other tests.

    """
    _test_mxfp_conv2d_eager_cpu(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
        frobenius_threshold=0.08,
        cosine_threshold=0.995,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp4_conv2d_eager_cpu(
    test_data: Callable[[], tuple[Conv2d, bool]],
) -> None:
    _test_mxfp_conv2d_eager_cpu(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
        frobenius_threshold=0.3,
        cosine_threshold=0.95,
    )
