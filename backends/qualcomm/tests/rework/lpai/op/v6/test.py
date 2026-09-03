# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from functools import partial
from pathlib import Path

import pytest

from executorch.backends.qualcomm.tests.rework.conftest import (
    check_exception,
    EXCEPTION_EXIR_PROGRAM,
    EXCEPTION_FROM_PASSES,
    EXPECT_NOT_ANNOTATED,
    EXPECT_NOT_FULLY_DELEGATED,
    Tolerance,
)
from executorch.backends.qualcomm.tests.rework.src.op import *  # noqa: F403
from executorch.backends.qualcomm.tests.rework.lpai.conftest import (
    enumerate_activation_dtype,
    with_lpai_context,
)


# e.g. get 68 from ".../rework/htp/unit_test/op/v68/test.py"
LPAI_ARCH = int(re.search(r".*v([0-9]+)$", Path(__file__).parent.name).group(1))
with_lpai_context = partial(with_lpai_context, hw_arch=LPAI_ARCH)


# abs not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_abs(request, kwargs):
    Abs.test(request, kwargs)  # noqa: F405


# acos not in lpai_rules but will be decomposed into equivalent ops
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_acos(request, kwargs):
    ACos.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_adaptive_avg_pool_1d_unsupported_io_shape(request, kwargs):
    AdaptiveAvgPool.test_1d_unsupported_io_shape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_adaptive_avg_pool_1d(request, kwargs):
    AdaptiveAvgPool.test_1d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_adaptive_avg_pool_2d_unsupported_io_shape(request, kwargs):
    AdaptiveAvgPool.test_2d_unsupported_io_shape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_adaptive_avg_pool_2d(request, kwargs):
    AdaptiveAvgPool.test_2d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_adaptive_avg_pool_3d_unsupported_io_shape(request, kwargs):
    AdaptiveAvgPool.test_3d_unsupported_io_shape(request, kwargs)  # noqa: F405


# adaptive_avg_pool3d not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_adaptive_avg_pool_3d(request, kwargs):
    AdaptiveAvgPool.test_3d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_adaptive_max_pool_2d(request, kwargs):
    AdaptiveMaxPool.test_2d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_adaptive_max_pool_2d_with_indices(request, kwargs):
    AdaptiveMaxPool.test_2d_with_indices(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_add(request, kwargs):
    Add.test(request, kwargs)  # noqa: F405


# addmm decomposes to mm+add before annotation; both in lpai_rules
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_addmm(request, kwargs):
    AddMM.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_alias(request, kwargs):
    Alias.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_amax(request, kwargs):
    AMax.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_amin(request, kwargs):
    AMin.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_any(request, kwargs):
    Any.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_arange_dtype_int(request, kwargs):
    Arange.test_dtype_int(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_arange_dtype_float(request, kwargs):
    Arange.test_dtype_float(request, kwargs)  # noqa: F405


# int64 cast for indices is not supported by lpai
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_argmax(request, kwargs):
    ArgMax.test(request, kwargs)  # noqa: F405


# int64 cast for indices is not supported by lpai
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_argmin(request, kwargs):
    ArgMin.test(request, kwargs)  # noqa: F405


# asin not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_asin(request, kwargs):
    ASin.test(request, kwargs)  # noqa: F405


# some decomposed ops are not supported by lpai
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_atan(request, kwargs):
    ATan.test(request, kwargs)  # noqa: F405


# some decomposed ops are not supported by lpai
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_atan2(request, kwargs):
    ATan2.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_avgpool_1d(request, kwargs):
    AvgPool.test_1d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_avgpool_2d(request, kwargs):
    AvgPool.test_2d(request, kwargs)  # noqa: F405


# avg_pool3d not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_avgpool_3d(request, kwargs):
    AvgPool.test_3d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_batchnorm_2d(request, kwargs):
    BatchNorm2d.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_bitwise_and_numeric(request, kwargs):
    BitwiseOp.test_and_numeric(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_bitwise_and_bool(request, kwargs):
    BitwiseOp.test_and_bool(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_bitwise_or_numeric(request, kwargs):
    BitwiseOp.test_or_numeric(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_bitwise_or_bool(request, kwargs):
    BitwiseOp.test_or_bool(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_bitwise_xor_numeric(request, kwargs):
    BitwiseOp.test_xor_numeric(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_bitwise_xor_bool(request, kwargs):
    BitwiseOp.test_xor_bool(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_bmm(request, kwargs):
    Bmm.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_cast(request, kwargs):
    Cast.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_cat(request, kwargs):
    Cat.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_cdist(request, kwargs):
    CDist.test(request, kwargs)  # noqa: F405


# ceil not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_ceil(request, kwargs):
    Ceil.test(request, kwargs)  # noqa: F405


# channel_shuffle not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_channel_shuffle(request, kwargs):
    ChannelShuffle.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_chunk(request, kwargs):
    Chunk.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_clamp(request, kwargs):
    Clamp.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_clamp_max(request, kwargs):
    ClampMax.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_clamp_min(request, kwargs):
    ClampMin.test(request, kwargs)  # noqa: F405


# clone not in lpai_rules but will be omitted
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_clone(request, kwargs):
    Clone.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": False, "expected": Tolerance()},
            id="8a8w_ptq",
        ),
        pytest.param(
            {"act": 8, "param": 8, "pcq": True, "expected": Tolerance()},
            id="8a8w_pcq",
        ),
    ],
)
@with_lpai_context
def test_conv1d(request, kwargs):
    Conv.test_1d(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": False, "expected": Tolerance()},
            id="8a8w_ptq",
        ),
        pytest.param(
            {"act": 8, "param": 8, "pcq": True, "expected": Tolerance()},
            id="8a8w_pcq",
        ),
    ],
)
@with_lpai_context
def test_conv1d_transpose(request, kwargs):
    Conv.test_1d_transpose(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": False, "expected": Tolerance()},
            id="8a8w_ptq",
        ),
        pytest.param(
            {"act": 8, "param": 8, "pcq": True, "expected": Tolerance()},
            id="8a8w_pcq",
        ),
    ],
)
@with_lpai_context
def test_conv2d(request, kwargs):
    Conv.test_2d(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": False, "expected": Tolerance()},
            id="8a8w_ptq",
        ),
        pytest.param(
            {"act": 8, "param": 8, "pcq": True, "expected": Tolerance()},
            id="8a8w_pcq",
        ),
    ],
)
@with_lpai_context
def test_conv2d_transpose(request, kwargs):
    Conv.test_2d_transpose(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": True, "expected": Tolerance()},
            id="8a8w_pcq",
        ),
    ],
)
@with_lpai_context
def test_conv2d_linear_like(request, kwargs):
    Conv.test_2d_linear_like(request, kwargs)  # noqa: F405


# conv3d not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_conv3d(request, kwargs):
    Conv.test_3d(request, kwargs)  # noqa: F405


# conv3d_transpose not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_conv3d_transpose(request, kwargs):
    Conv.test_3d_transpose(request, kwargs)  # noqa: F405


# cos not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_cos(request, kwargs):
    Cos.test(request, kwargs)  # noqa: F405


# cumsum not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_cumsum(request, kwargs):
    CumSum.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_div(request, kwargs):
    Div.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_div_with_rounding_mode(request, kwargs):
    DivWithRoundingMode.test(request, kwargs)  # noqa: F405


# einsum decomposes to bmm/matmul before annotation; both in lpai_rules
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_einsum(request, kwargs):
    Einsum.test(request, kwargs)  # noqa: F405


# elu not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_elu(request, kwargs):
    Elu.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": False, "expected": Tolerance()},
            id="8a8w_ptq",
        ),
    ],
)
@with_lpai_context
def test_embedding(request, kwargs):
    Embedding.test(request, kwargs)  # noqa: F405


# equal not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_equal(request, kwargs):
    Equal.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_exp(request, kwargs):
    Exp.test(request, kwargs)  # noqa: F405


# expand not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_expand(request, kwargs):
    Expand.test(request, kwargs)  # noqa: F405


# expand_as not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_expand_as(request, kwargs):
    ExpandAs.test(request, kwargs)  # noqa: F405


# expm1 not in lpai_rules but will be decomposed
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_expm1(request, kwargs):
    ExpM1.test(request, kwargs)  # noqa: F405


# fill translates to static tensor in QNN; backend-agnostic
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_fill(request, kwargs):
    Fill.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_flip(request, kwargs):
    Flip.test(request, kwargs)  # noqa: F405


# floor not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_floor(request, kwargs):
    Floor.test(request, kwargs)  # noqa: F405


# floor_divide not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_floor_divide(request, kwargs):
    FloorDivide.test(request, kwargs)  # noqa: F405


# fold uses col2im which is in lpai_rules (ColIm, qnn_op=None)
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_fold(request, kwargs):
    Fold.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
    ]
)
@with_lpai_context
def test_fold_unsupported_parameters(request, kwargs):
    Fold.test_unsupported_parameters(request, kwargs)  # noqa: F405


# full/full_like translate to static tensors in QNN; backend-agnostic
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_full(request, kwargs):
    Full.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_full_like(request, kwargs):
    FullLike.test(request, kwargs)  # noqa: F405


# gather is in lpai_rules (Embedding class handles index/gather/index_select)
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_gather(request, kwargs):
    Gather.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_gelu(request, kwargs):
    Gelu.test(request, kwargs)  # noqa: F405


# glu decomposes to chunk+sigmoid+mul before annotation; all in lpai_rules
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_glu(request, kwargs):
    Glu.test(request, kwargs)  # noqa: F405


# greater not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_greater(request, kwargs):
    Greater.test_gt(request, kwargs)  # noqa: F405


# greater_equal not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_greater_equal(request, kwargs):
    Greater.test_ge(request, kwargs)  # noqa: F405


# grid_sample not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_grid_sample_4d(request, kwargs):
    GridSample.test_4d(request, kwargs)  # noqa: F405


# grid_sample not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_grid_sample_5d(request, kwargs):
    GridSample.test_5d(request, kwargs)  # noqa: F405


# group_norm not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_group_norm(request, kwargs):
    GroupNorm.test(request, kwargs)  # noqa: F405


# hardsigmoid: DecomposeHardsigmoid runs before annotation; decomposed ops in lpai_rules
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_hardsigmoid(request, kwargs):
    HardSigmoid.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_hardswish(request, kwargs):
    HardSwish.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_hardtanh(request, kwargs):
    HardTanh.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_index(request, kwargs):
    Index.test(request, kwargs)  # noqa: F405


# decomposed ops might not be supported by lpai
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_index_copy(request, kwargs):
    IndexCopy.test(request, kwargs)  # noqa: F405


# decomposed ops might not be supported by lpai
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_index_put(request, kwargs):
    IndexPut.test(request, kwargs)  # noqa: F405


# decomposed ops might not be supported by lpai
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_index_select(request, kwargs):
    IndexSelect.test(request, kwargs)  # noqa: F405


# instance_norm not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_instance_norm_2d(request, kwargs):
    InstanceNorm2d.test(request, kwargs)  # noqa: F405


# registered by the partitioner to not be decomposed but failed to be delegated
@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
    ]
)
@with_lpai_context
def test_interpolate_bicubic(request, kwargs):
    Interpolate.test_bicubic(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_interpolate_bilinear(request, kwargs):
    Interpolate.test_bilinear(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_interpolate_nearest(request, kwargs):
    Interpolate.test_nearest(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_is_inf(request, kwargs):
    IsInf.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_is_nan(request, kwargs):
    IsNan.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_layer_norm(request, kwargs):
    LayerNorm.test(request, kwargs)  # noqa: F405


# maps to prelu
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_leaky_relu(request, kwargs):
    LeakyReLU.test(request, kwargs)  # noqa: F405


# less_equal not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_less_equal(request, kwargs):
    LessEqual.test(request, kwargs)  # noqa: F405


# less_than not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_less_than(request, kwargs):
    LessThan.test(request, kwargs)  # noqa: F405


# decomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_linalg_vector_norm(request, kwargs):
    LinalgVectorNorm.test(request, kwargs)  # noqa: F405


# LPBQ not applicable to LPAI v6 (requires HTP V69+ feature)
@pytest.mark.skip(reason="LPBQ quantization is not supported on LPAI v6")
@with_lpai_context
def test_linear_block_quant(request, kwargs):
    Linear.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": False, "expected": Tolerance()},
            id="8a8w_ptq",
        ),
        pytest.param(
            {"act": 8, "param": 8, "pcq": True, "expected": Tolerance()},
            id="8a8w_pcq",
        ),
    ],
)
@with_lpai_context
def test_linear_general(request, kwargs):
    Linear.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 8, "param": 8, "pcq": False, "expected": Tolerance()},
            id="8a8w_ptq",
        ),
    ],
)
@with_lpai_context
def test_linear_non_constant_weight(request, kwargs):
    LinearNonConstantWeight.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_log(request, kwargs):
    Log.test(request, kwargs)  # noqa: F405


# log10 not in lpai_rules but will be decomposed
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_log10(request, kwargs):
    Log10.test(request, kwargs)  # noqa: F405


# log1p not in lpai_rules but will be decomposed
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_log1p(request, kwargs):
    Log1p.test(request, kwargs)  # noqa: F405


# log2 not in lpai_rules but will be decomposed
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_log2(request, kwargs):
    Log2.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_log_softmax(request, kwargs):
    LogSoftmax.test(request, kwargs)  # noqa: F405


# logical_and not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_logical_and(request, kwargs):
    LogicalAnd.test(request, kwargs)  # noqa: F405


# logical_not not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_logical_not(request, kwargs):
    LogicalNot.test(request, kwargs)  # noqa: F405


# decomposed ops are not supported with invalid weight fallback triggered
@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
    ]
)
@with_lpai_context
def test_masked_fill(request, kwargs):
    MaskedFill.test(request, kwargs)  # noqa: F405


# cast op for indices is not supported
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_max_dim(request, kwargs):
    MaxDim.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_maximum(request, kwargs):
    Maximum.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_maxpool_2d(request, kwargs):
    MaxPool2d.test(request, kwargs)  # noqa: F405


# max_pool3d not in lpai_rules and the decomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_maxpool_3d(request, kwargs):
    MaxPool3d.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_mean(request, kwargs):
    Mean.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_mha(request, kwargs):
    MultiheadAttention.test(request, kwargs)  # noqa: F405


# cast op for indices is not supported
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_min_dim(request, kwargs):
    MinDim.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_minimum(request, kwargs):
    Minimum.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_narrow(request, kwargs):
    Narrow.test(request, kwargs)  # noqa: F405


# neg not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_neg(request, kwargs):
    Neg.test(request, kwargs)  # noqa: F405


# not_equal not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_not_equal(request, kwargs):
    NotEqual.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_pad_constant(request, kwargs):
    Pad.test_constant(request, kwargs)  # noqa: F405


# registered by the partitioner to not be decomposed but failed to be delegated
@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
    ]
)
@with_lpai_context
def test_pad_reflect(request, kwargs):
    Pad.test_reflect(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_permute(request, kwargs):
    Permute.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_pixel_shuffle(request, kwargs):
    PixelShuffle.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_pixel_unshuffle(request, kwargs):
    PixelUnshuffle.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_pow_scalar(request, kwargs):
    PowScalar.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_pow_tensor_scalar(request, kwargs):
    PowTensorScalar.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_prelu(request, kwargs):
    PReLU.test(request, kwargs)  # noqa: F405


# rand not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_rand(request, kwargs):
    Rand.test(request, kwargs)  # noqa: F405


# reciprocal: DecomposeReciprocal decomposes to div(1, x); div is in lpai_rules
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_reciprocal(request, kwargs):
    Reciprocal.test(request, kwargs)  # noqa: F405


# registered by the partitioner to not be decomposed but failed to be delegated
@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
    ]
)
@with_lpai_context
def test_reflection_pad_1d(request, kwargs):
    ReflectionPad.test_3d(request, kwargs)  # noqa: F405


# registered by the partitioner to not be decomposed but failed to be delegated
@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
    ]
)
@with_lpai_context
def test_reflection_pad_2d(request, kwargs):
    ReflectionPad.test_4d(request, kwargs)  # noqa: F405


# reflection_pad3d not in lpai_rules but will be decomposed
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_reflection_pad_3d(request, kwargs):
    ReflectionPad.test_5d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_relu(request, kwargs):
    Relu.test(request, kwargs)  # noqa: F405


# relu6 decomposes to hardtanh(0, 6); hardtanh is in lpai_rules (ReluMinMax)
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_relu6(request, kwargs):
    Relu6.test(request, kwargs)  # noqa: F405


# decomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_remainder(request, kwargs):
    Remainder.test(request, kwargs)  # noqa: F405


# repeat not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_repeat(request, kwargs):
    Repeat.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_reshape_2d_to_4d_random_reshape(request, kwargs):
    Reshape.test_2d_to_4d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_reshape_2d_to_4d_flatten_last_two_dims(request, kwargs):
    Reshape.test_2d_to_4d_flatten_last_two_dims(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_reshape_5d_random_reshape(request, kwargs):
    Reshape.test_5d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_reshape_5d_flatten_last_two_dims(request, kwargs):
    Reshape.test_5d_flatten_last_two_dims(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_rms_norm(request, kwargs):
    RmsNorm.test(request, kwargs)  # noqa: F405


# roll not in lpai_rules but decomposed ops are fully delegated
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_roll(request, kwargs):
    Roll.test(request, kwargs)  # noqa: F405


# round not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_round(request, kwargs):
    Round.test(request, kwargs)  # noqa: F405


# rsqrt not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_rsqrt(request, kwargs):
    Rsqrt.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_sdpa(request, kwargs):
    ScaledDotProductAttention.test(request, kwargs)  # noqa: F405


# scatter.src not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_scatter_src(request, kwargs):
    ScatterSrc.test(request, kwargs)  # noqa: F405


# select_copy maps to aten.select.int which is in lpai_rules (StrideSlice)
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_select_copy(request, kwargs):
    SelectCopy.test(request, kwargs)  # noqa: F405


# select_scatter not in lpai_rules and decomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_select_scatter(request, kwargs):
    SelectScatter.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_sigmoid(request, kwargs):
    Sigmoid.test(request, kwargs)  # noqa: F405


# sign not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_sign(request, kwargs):
    Sign.test(request, kwargs)  # noqa: F405


# sin not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_sin(request, kwargs):
    Sin.test(request, kwargs)  # noqa: F405


# slice_copy maps to aten.slice.Tensor which is in lpai_rules (StrideSlice)
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_slice_copy(request, kwargs):
    SliceCopy.test(request, kwargs)  # noqa: F405


# slice_scatter not in lpai_rules and decomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_slice_scatter(request, kwargs):
    SliceScatter.test(request, kwargs)  # noqa: F405


# scatter not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_scatter_value(request, kwargs):
    ScatterValue.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_softmax(request, kwargs):
    Softmax.test(request, kwargs)  # noqa: F405


# sort not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_sort(request, kwargs):
    Sort.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_split(request, kwargs):
    Split.test(request, kwargs)  # noqa: F405


# square is in lpai_rules (Pow class handles square.default)
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_square(request, kwargs):
    Square.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_squeeze(request, kwargs):
    Squeeze.test(request, kwargs)  # noqa: F405


# stack maps to OpPack which is HTP-specific
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_stack(request, kwargs):
    Stack.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_sum_int_list(request, kwargs):
    SumIntList.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_swapaxes(request, kwargs):
    SwapAxes.test(request, kwargs)  # noqa: F405


# tan not in lpai_rules and the decomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_tan(request, kwargs):
    Tan.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_tanh(request, kwargs):
    Tanh.test(request, kwargs)  # noqa: F405


# threshold not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
    ]
)
@with_lpai_context
def test_threshold(request, kwargs):
    Threshold.test(request, kwargs)  # noqa: F405


# triu not in lpai_rulesdecomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_triu(request, kwargs):
    Triu.test(request, kwargs)  # noqa: F405


# triu not in lpai_rulesdecomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_triu_constant(request, kwargs):
    Triu.test_constant(request, kwargs)  # noqa: F405


# trunc not in lpai_rules and decomposed ops are not fully delegated
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_trunc(request, kwargs):
    Trunc.test(request, kwargs)  # noqa: F405


# topk not in lpai_rules, use EXPECT_NOT_FULLY_DELEGATED for there are
# other ops in the test body
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_topk(request, kwargs):
    TopK.test(request, kwargs)  # noqa: F405


# unbind maps to OpUnpack which is HTP-specific
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_unbind(request, kwargs):
    Unbind.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_unflatten(request, kwargs):
    Unflatten.test(request, kwargs)  # noqa: F405


# unfold uses im2col which is in lpai_rules (ColIm, qnn_op=None)
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_unfold(request, kwargs):
    Unfold.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
    ]
)
@with_lpai_context
def test_unfold_unsupported(request, kwargs):
    Unfold.test_unsupported(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_unsqueeze(request, kwargs):
    Unsqueeze.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_view_2d_to_4d_random_reshape(request, kwargs):
    View.test_2d_to_4d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_view_2d_to_4d_flatten_last_two_dims(request, kwargs):
    View.test_2d_to_4d_flatten_last_two_dims(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_view_5d_random_reshape(request, kwargs):
    View.test_5d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_view_5d_flatten_last_two_dims(request, kwargs):
    View.test_5d_flatten_last_two_dims(request, kwargs)  # noqa: F405


# where not in lpai_rules
@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_lpai_context
def test_where(request, kwargs):
    Where.test(request, kwargs)  # noqa: F405


# var not in lpai_rules but will be decomposed
@enumerate_activation_dtype([Tolerance()])
@with_lpai_context
def test_var(request, kwargs):
    Var.test(request, kwargs)  # noqa: F405
