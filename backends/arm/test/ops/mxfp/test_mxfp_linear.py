# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
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
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2

aten_op = "torch.ops.tosa_mxfp.linear.default"

input_t1 = Tuple[torch.Tensor]


def _block_input_rank1() -> torch.Tensor:
    """Create a rank-1 input with distinct MXFP activation block scales."""

    return torch.cat(
        (
            1e-3 * torch.randn(32),
            100.0 * torch.randn(32),
        )
    )


def _block_input_rank2() -> torch.Tensor:
    """Create a rank-2 input with per-row activation block scale changes."""

    return torch.stack(
        (
            _block_input_rank1(),
            torch.cat(
                (
                    100.0 * torch.randn(32),
                    1e-3 * torch.randn(32),
                )
            ),
        )
    )


def _channels_last_rank4_input() -> torch.Tensor:
    """Create a rank-4 input with channels-last dim order."""

    return torch.rand(1, 2, 2, 64).to(memory_format=torch.channels_last)


_test_data_rank1_fp = {
    "mxfp_linear_rank1_zeros": lambda: (
        torch.zeros(32 * 8),
        5,
        True,
        False,
    ),
    "mxfp_linear_rank1_rand": lambda: (
        torch.rand(32),
        16,
        False,
        False,
    ),
}

_test_data_rank2_fp = {
    "mxfp_linear_rank2_zeros": lambda: (
        torch.zeros(4, 32),
        16,
        True,
        False,
    ),
    "mxfp_linear_rank2_rand": lambda: (
        torch.rand(4, 32 * 6),
        13,
        True,
        False,
    ),
}

_test_data_rank3_fp = {
    "mxfp_linear_rank3_zeros": lambda: (
        torch.zeros(2, 4, 32 * 3),
        1,
        True,
        False,
    ),
    "mxfp_linear_rank3_rand": lambda: (
        torch.rand(2, 4, 32),
        20,
        True,
        False,
    ),
}

_test_data_rank4_fp = {
    "mxfp_linear_rank4_zeros": lambda: (
        torch.zeros(2, 3, 4, 32 * 24),
        8,
        True,
        False,
    ),
    "mxfp_linear_rank4_rand": lambda: (
        torch.rand(2, 3, 4, 32 * 32),
        64,
        False,
        False,
    ),
}

_test_data_block_fp = {
    "mxfp_linear_rank1_block_weights": lambda: (
        torch.ones(64),
        4,
        False,
        True,
    ),
    "mxfp_linear_rank1_block_weights_block_activations": lambda: (
        _block_input_rank1(),
        4,
        False,
        True,
    ),
    "mxfp_linear_rank2_block_weights_block_activations": lambda: (
        _block_input_rank2(),
        4,
        False,
        True,
    ),
}

_test_data_dim_order_fp = {
    "mxfp_linear_rank4_channels_last": lambda: (
        _channels_last_rank4_input(),
        8,
        True,
        False,
    ),
}

test_data_fp = (
    _test_data_rank1_fp
    | _test_data_rank2_fp
    | _test_data_rank3_fp
    | _test_data_rank4_fp
    | _test_data_block_fp
    | _test_data_dim_order_fp
)

test_data_vgf_fp = test_data_fp

# TODO: MLETORCH-2141
_vgf_xfail_reason = (
    "MXFP is not yet supported in the VGF toolchain. Enable this test when "
    "toolchain support is available."
)
_vgf_xfails: dict[str, str | tuple[str, type[Exception]]] = {
    test_case: _vgf_xfail_reason for test_case in test_data_vgf_fp
}


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 8,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def set_block_test_weights(self) -> None:
        """Set weights to exercise separate MXFP weight block scales.

        The first two logical 32-wide input blocks use different magnitudes so
        tests can verify block scaling does not share one scale across blocks.

        """
        if self.fc.weight.shape[1] < 64:
            raise ValueError(
                "Block test weights require at least 64 input features (2 blocks), got "
                f"{tuple(self.fc.weight.shape)}"
            )

        with torch.no_grad():
            self.fc.weight.zero_()
            for row in range(self.fc.weight.shape[0]):
                # Small values in the first block.
                self.fc.weight[row, 0:32] = 1e-3
                # Large values in the next block to require a different scale.
                self.fc.weight[row, 32:64] = 100.0
            if self.fc.bias is not None:
                self.fc.bias.zero_()


def _is_linear(module: torch.nn.Module, _fqn: str) -> bool:
    return isinstance(module, torch.nn.Linear)


def _test_mxfp_linear_eager_cpu(
    test_data,
    config: MXFPOpConfig,
    frobenius_threshold=0.3,
    cosine_threshold=0.95,
) -> None:
    test_input, out_features, has_bias, set_block_weights = test_data()
    in_features = test_input.shape[-1]
    ref_model = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
    ).eval()

    if set_block_weights:
        ref_model.set_block_test_weights()
    test_model = copy.deepcopy(ref_model).eval()

    to_mxfp(test_model, config, filter_fn=_is_linear)

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


def _test_mxfp_linear_vgf(
    test_data,
    config: MXFPOpConfig,
    frobenius_threshold,
    cosine_threshold,
) -> None:
    test_input, out_features, has_bias, set_block_weights = test_data()
    in_features = test_input.shape[-1]
    module = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
    ).eval()

    if set_block_weights:
        module.set_block_test_weights()

    pipeline = MXFPVgfPipeline[input_t1](
        module,
        (test_input,),
        aten_op,
        filter_fn=_is_linear,
        frobenius_threshold=frobenius_threshold,
        cosine_threshold=cosine_threshold,
        mxfp_config=config,
        tosa_spec="TOSA-1.1+FP+mxfp",
    )
    pipeline.run()


def _test_mxfp_linear_tosa_FP(
    test_data,
    config: MXFPOpConfig,
    frobenius_threshold=0.08,
    cosine_threshold=0.995,
) -> None:
    test_input, out_features, has_bias, set_block_weights = test_data()
    in_features = test_input.shape[-1]
    module = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
    ).eval()

    if set_block_weights:
        module.set_block_test_weights()

    pipeline = MXFPTosaPipelineFP[input_t1](
        module,
        (test_input,),
        aten_op,
        filter_fn=_is_linear,
        frobenius_threshold=frobenius_threshold,
        cosine_threshold=cosine_threshold,
        mxfp_config=config,
        tosa_version="1.1",
        tosa_extensions=["mxfp"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp)
def test_mxfp8_linear_tosa_FP(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_tosa_FP(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp4_linear_tosa_FP(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_tosa_FP(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
        frobenius_threshold=0.3,
        cosine_threshold=0.95,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp6_e2m3_linear_tosa_FP(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_tosa_FP(
        test_data,
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E2M3),
        frobenius_threshold=0.2,
        cosine_threshold=0.98,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp6_e3m2_linear_tosa_FP(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_tosa_FP(
        test_data,
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E3M2),
        frobenius_threshold=0.2,
        cosine_threshold=0.98,
    )


@common.parametrize("test_data", test_data_vgf_fp, xfails=_vgf_xfails)
@common.SkipIfNoModelConverter
def test_mxfp8_linear_vgf(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_vgf(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
        frobenius_threshold=0.08,
        cosine_threshold=0.995,
    )


@common.parametrize("test_data", test_data_vgf_fp, xfails=_vgf_xfails)
@common.SkipIfNoModelConverter
def test_mxfp4_linear_vgf(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_vgf(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
        frobenius_threshold=0.3,
        cosine_threshold=0.95,
    )


@common.parametrize("test_data", test_data_vgf_fp, xfails=_vgf_xfails)
@common.SkipIfNoModelConverter
def test_mxfp6_e2m3_linear_vgf(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_vgf(
        test_data,
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E2M3),
        frobenius_threshold=0.2,
        cosine_threshold=0.98,
    )


@common.parametrize("test_data", test_data_vgf_fp, xfails=_vgf_xfails)
@common.SkipIfNoModelConverter
def test_mxfp6_e3m2_linear_vgf(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_vgf(
        test_data,
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E3M2),
        frobenius_threshold=0.2,
        cosine_threshold=0.98,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp8_linear_eager_cpu(test_data: torch.Tensor) -> None:
    """Check eager MXFP implementation.

    The Arm lowering tests compare lowered output against the eager CPU
    implementation, so the eager implementation must be accurate for it to be
    used as a reference in other tests.

    """
    _test_mxfp_linear_eager_cpu(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
        frobenius_threshold=0.08,
        cosine_threshold=0.995,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp4_linear_eager_cpu(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_eager_cpu(
        test_data,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
        frobenius_threshold=0.3,
        cosine_threshold=0.95,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp6_e2m3_linear_eager_cpu(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_eager_cpu(
        test_data,
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E2M3),
        frobenius_threshold=0.2,
        cosine_threshold=0.98,
    )


@common.parametrize("test_data", test_data_fp)
def test_mxfp6_e3m2_linear_eager_cpu(test_data: torch.Tensor) -> None:
    _test_mxfp_linear_eager_cpu(
        test_data,
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E3M2),
        frobenius_threshold=0.2,
        cosine_threshold=0.98,
    )
