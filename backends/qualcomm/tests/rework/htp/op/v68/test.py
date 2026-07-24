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
    CosineSimilarity,
    EXCEPTION_EXIR_PROGRAM,
    EXCEPTION_FROM_PASSES,
    EXPECT_NOT_ANNOTATED,
    EXPECT_NOT_FULLY_DELEGATED,
    SkipOutputCheck,
    Tolerance,
)
from executorch.backends.qualcomm.tests.rework.src.op import *  # noqa: F403
from executorch.backends.qualcomm.tests.rework.htp.conftest import (
    enumerate_activation_dtype,
    with_htp_context,
)

# e.g. get 68 from ".../rework/htp/unit_test/op/v68/test.py"
HTP_ARCH = int(re.search(r".*v([0-9]+)$", Path(__file__).parent.name).group(1))
with_htp_context = partial(with_htp_context, hw_arch=HTP_ARCH)


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_abs(request, kwargs):
    Abs.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_acos(request, kwargs):
    ACos.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
    ]
)
@with_htp_context
def test_adaptive_avg_pool_1d_unsupported_io_shape(request, kwargs):
    AdaptiveAvgPool.test_1d_unsupported_io_shape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_adaptive_avg_pool_1d(request, kwargs):
    AdaptiveAvgPool.test_1d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_EXIR_PROGRAM)),
    ]
)
@with_htp_context
def test_adaptive_avg_pool_2d_unsupported_io_shape(request, kwargs):
    AdaptiveAvgPool.test_2d_unsupported_io_shape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_adaptive_avg_pool_2d(request, kwargs):
    AdaptiveAvgPool.test_2d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_adaptive_avg_pool_3d_unsupported_io_shape(request, kwargs):
    AdaptiveAvgPool.test_3d_unsupported_io_shape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_adaptive_avg_pool_3d(request, kwargs):
    AdaptiveAvgPool.test_3d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_adaptive_max_pool_2d(request, kwargs):
    AdaptiveMaxPool.test_2d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_adaptive_max_pool_2d_with_indices(request, kwargs):
    AdaptiveMaxPool.test_2d_with_indices(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_add(request, kwargs):
    Add.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        Tolerance(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_addmm(request, kwargs):
    AddMM.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_alias(request, kwargs):
    Alias.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_amax(request, kwargs):
    AMax.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_amin(request, kwargs):
    AMin.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_any(request, kwargs):
    Any.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_arange_dtype_int(request, kwargs):
    Arange.test_dtype_int(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_arange_dtype_float(request, kwargs):
    Arange.test_dtype_float(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_argmax(request, kwargs):
    ArgMax.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_argmin(request, kwargs):
    ArgMin.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        Tolerance(),
        Tolerance(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_asin(request, kwargs):
    ASin.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_atan(request, kwargs):
    ATan.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        Tolerance(),
        Tolerance(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_atan2(request, kwargs):
    ATan2.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_avgpool_1d(request, kwargs):
    AvgPool.test_1d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_avgpool_2d(request, kwargs):
    AvgPool.test_2d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_avgpool_3d(request, kwargs):
    AvgPool.test_3d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_batchnorm_2d(request, kwargs):
    BatchNorm2d.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_bitwise_and_numeric(request, kwargs):
    BitwiseOp.test_and_numeric(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_bitwise_and_bool(request, kwargs):
    BitwiseOp.test_and_bool(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_bitwise_or_numeric(request, kwargs):
    BitwiseOp.test_or_numeric(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_bitwise_or_bool(request, kwargs):
    BitwiseOp.test_or_bool(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_bitwise_xor_numeric(request, kwargs):
    BitwiseOp.test_xor_numeric(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_bitwise_xor_bool(request, kwargs):
    BitwiseOp.test_xor_bool(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        Tolerance(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_bmm(request, kwargs):
    Bmm.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_cast(request, kwargs):
    Cast.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_cat(request, kwargs):
    Cat.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_cdist(request, kwargs):
    CDist.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_ceil(request, kwargs):
    Ceil.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_channel_shuffle(request, kwargs):
    ChannelShuffle.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_chunk(request, kwargs):
    Chunk.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_clamp(request, kwargs):
    Clamp.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_clamp_max(request, kwargs):
    ClampMax.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_clamp_min(request, kwargs):
    ClampMin.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_clone(request, kwargs):
    Clone.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": act, "param": param, "pcq": pcq, "expected": expected},
            id=id,
        )
        for act, param, pcq, expected, id in [
            (8, 8, True, Tolerance(), "8a8w_pcq"),
            (16, 4, True, CosineSimilarity(0.95), "16a4w_pcq"),
            (None, None, False, Tolerance(rtol=1e-1), "fp"),
        ]
    ],
)
@with_htp_context
def test_conv1d(request, kwargs):
    Conv.test_1d(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": act, "param": param, "pcq": pcq, "expected": expected},
            id=id,
        )
        for act, param, pcq, expected, id in [
            (8, 8, True, Tolerance(), "8a8w_pcq"),
            (16, 4, True, CosineSimilarity(0.95), "16a4w_pcq"),
            (None, None, False, Tolerance(rtol=1e-1), "fp"),
        ]
    ],
)
@with_htp_context
def test_conv1d_transpose(request, kwargs):
    Conv.test_1d_transpose(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": act, "param": param, "pcq": pcq, "expected": expected},
            id=id,
        )
        for act, param, pcq, expected, id in [
            (8, 8, True, Tolerance(), "8a8w_pcq"),
            (16, 4, True, CosineSimilarity(0.95), "16a4w_pcq"),
            (None, None, False, Tolerance(rtol=1e-1), "fp"),
        ]
    ],
)
@with_htp_context
def test_conv2d(request, kwargs):
    Conv.test_2d(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": act, "param": param, "pcq": pcq, "expected": expected},
            id=id,
        )
        for act, param, pcq, expected, id in [
            (8, 8, True, Tolerance(), "8a8w_pcq"),
            (16, 4, True, CosineSimilarity(0.95), "16a4w_pcq"),
            (None, None, False, Tolerance(rtol=1e-1), "fp"),
        ]
    ],
)
@with_htp_context
def test_conv2d_transpose(request, kwargs):
    Conv.test_2d_transpose(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "act": 16,
                "param": 4,
                "lpbq": True,
                "block_sz_map": {"conv2d": (1, 32, 1, 1)},
                "expected": pytest.raises(
                    AssertionError, match=EXPECT_NOT_FULLY_DELEGATED
                ),
            },
            id="16a4w_lpbq",
        ),
        pytest.param(
            {"act": "fp16", "param": 8, "pcq": True, "expected": Tolerance()},
            id="fp16a8w_pcq",
        ),
    ],
)
@with_htp_context
def test_conv2d_linear_like(request, kwargs):
    Conv.test_2d_linear_like(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": act, "param": param, "pcq": pcq, "expected": expected},
            id=id,
        )
        for act, param, pcq, expected, id in [
            # no bitwidth support for conv3d
            (8, 8, True, Tolerance(), "8a8w_pcq"),
            (None, None, False, Tolerance(rtol=1e-1), "fp"),
        ]
    ],
)
@with_htp_context
def test_conv3d(request, kwargs):
    Conv.test_3d(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": act, "param": param, "pcq": pcq, "expected": expected},
            id=id,
        )
        for act, param, pcq, expected, id in [
            # no bitwidth support for conv3d
            (8, 8, True, Tolerance(), "8a8w_pcq"),
            (None, None, False, Tolerance(rtol=1e-1), "fp"),
        ]
    ],
)
@with_htp_context
def test_conv3d_transpose(request, kwargs):
    Conv.test_3d_transpose(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_cos(request, kwargs):
    Cos.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_cumsum(request, kwargs):
    CumSum.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_div(request, kwargs):
    Div.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_div_with_rounding_mode(request, kwargs):
    DivWithRoundingMode.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        Tolerance(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_einsum(request, kwargs):
    Einsum.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_elu(request, kwargs):
    Elu.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": act, "param": param, "pcq": pcq, "expected": expected},
            id=id,
        )
        for act, param, pcq, expected, id in [
            (8, 8, False, Tolerance(), "8a8w_ptq"),
            (16, 16, False, Tolerance(), "16a16w_ptq"),
            (16, 8, True, Tolerance(), "16a8w_pcq"),
            (None, None, False, Tolerance(rtol=1e-1), "fp"),
        ]
    ],
)
@with_htp_context
def test_embedding(request, kwargs):
    Embedding.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_equal(request, kwargs):
    Equal.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_exp(request, kwargs):
    Exp.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_expand(request, kwargs):
    Expand.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_expand_as(request, kwargs):
    ExpandAs.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_expm1(request, kwargs):
    ExpM1.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_fill(request, kwargs):
    Fill.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_flip(request, kwargs):
    Flip.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_floor(request, kwargs):
    Floor.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_floor_divide(request, kwargs):
    FloorDivide.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_fold(request, kwargs):
    Fold.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
    ]
)
@with_htp_context
def test_fold_unsupported_parameters(request, kwargs):
    Fold.test_unsupported_parameters(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_full(request, kwargs):
    Full.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_full_like(request, kwargs):
    FullLike.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_gather(request, kwargs):
    Gather.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_gelu(request, kwargs):
    Gelu.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_glu(request, kwargs):
    Glu.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_greater(request, kwargs):
    Greater.test_gt(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_greater_equal(request, kwargs):
    Greater.test_ge(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_grid_sample_4d(request, kwargs):
    GridSample.test_4d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        CosineSimilarity(0.95),
        Tolerance(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_grid_sample_5d(request, kwargs):
    GridSample.test_5d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_group_norm(request, kwargs):
    GroupNorm.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_hardsigmoid(request, kwargs):
    HardSigmoid.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_hardswish(request, kwargs):
    HardSwish.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_hardtanh(request, kwargs):
    HardTanh.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_index(request, kwargs):
    Index.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_index_copy(request, kwargs):
    IndexCopy.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_index_put(request, kwargs):
    IndexPut.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_index_select(request, kwargs):
    IndexSelect.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_instance_norm_2d(request, kwargs):
    InstanceNorm2d.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_interpolate_bicubic(request, kwargs):
    Interpolate.test_bicubic(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_interpolate_bilinear(request, kwargs):
    Interpolate.test_bilinear(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_interpolate_nearest(request, kwargs):
    Interpolate.test_nearest(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_is_inf(request, kwargs):
    IsInf.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_ANNOTATED),
        Tolerance(),
    ]
)
@with_htp_context
def test_is_nan(request, kwargs):
    IsNan.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_layer_norm(request, kwargs):
    LayerNorm.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_leaky_relu(request, kwargs):
    LeakyReLU.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_less_equal(request, kwargs):
    LessEqual.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_less_than(request, kwargs):
    LessThan.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_linalg_vector_norm(request, kwargs):
    LinalgVectorNorm.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "act": 16,
                "param": 4,
                "pcq": False,
                "lpbq": True,
                "block_sz_map": {"linear": (1, 32)},
                "expected": Tolerance(),
            },
            id="16a4w_lpbq",
        ),
    ],
)
@with_htp_context
def test_linear_block_quant(request, kwargs):
    Linear.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"act": 16, "param": 16, "pcq": False, "expected": Tolerance()},
            id="16a16w_ptq",
        ),
        pytest.param(
            {"act": 8, "param": 8, "pcq": True, "expected": Tolerance()},
            id="8a8w_pcq",
        ),
        pytest.param(
            {"act": 16, "param": 4, "pcq": True, "expected": CosineSimilarity(0.95)},
            id="16a4w_pcq",
        ),
        pytest.param(
            {"act": 16, "param": 8, "pcq": True, "expected": Tolerance()},
            id="16a8w_pcq",
        ),
        pytest.param(
            {"act": "fp16", "param": 8, "pcq": True, "expected": Tolerance()},
            id="fp16a8w_pcq",
        ),
        pytest.param(
            {"act": 16, "param": 2, "pcq": True, "expected": CosineSimilarity(0.9)},
            id="16a2w_pcq",
        ),
        pytest.param(
            {
                "act": None,
                "param": None,
                "pcq": False,
                "expected": Tolerance(rtol=1e-1),
            },
            id="fp",
        ),
    ],
)
@with_htp_context
def test_linear_general(request, kwargs):
    Linear.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {
                "act": 16,
                "param": 16,
                "pcq": False,
                "expected": pytest.raises(AssertionError, match=Tolerance()),
            },
            id="16a16w_ptq",
        ),
        pytest.param(
            {
                "act": None,
                "param": None,
                "pcq": False,
                "expected": Tolerance(rtol=1e-1),
            },
            id="fp",
        ),
    ],
)
@with_htp_context
def test_linear_non_constant_weight(request, kwargs):
    LinearNonConstantWeight.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_log(request, kwargs):
    Log.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_log10(request, kwargs):
    Log10.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_log1p(request, kwargs):
    Log1p.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_log2(request, kwargs):
    Log2.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_log_softmax(request, kwargs):
    LogSoftmax.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_logical_and(request, kwargs):
    LogicalAnd.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_logical_not(request, kwargs):
    LogicalNot.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_masked_fill(request, kwargs):
    MaskedFill.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_max_dim(request, kwargs):
    MaxDim.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_maximum(request, kwargs):
    Maximum.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_maxpool_2d(request, kwargs):
    MaxPool2d.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_maxpool_3d(request, kwargs):
    MaxPool3d.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_mean(request, kwargs):
    Mean.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_mha(request, kwargs):
    MultiheadAttention.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_min_dim(request, kwargs):
    MinDim.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_minimum(request, kwargs):
    Minimum.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_narrow(request, kwargs):
    Narrow.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_neg(request, kwargs):
    Neg.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_not_equal(request, kwargs):
    NotEqual.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_pad_constant(request, kwargs):
    Pad.test_constant(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_pad_reflect(request, kwargs):
    Pad.test_reflect(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_permute(request, kwargs):
    Permute.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_pixel_shuffle(request, kwargs):
    PixelShuffle.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_pixel_unshuffle(request, kwargs):
    PixelUnshuffle.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_pow_scalar(request, kwargs):
    PowScalar.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_pow_tensor_scalar(request, kwargs):
    PowTensorScalar.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_prelu(request, kwargs):
    PReLU.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        SkipOutputCheck(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
    ]
)
@with_htp_context
def test_rand(request, kwargs):
    Rand.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_reciprocal(request, kwargs):
    Reciprocal.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_reflection_pad_1d(request, kwargs):
    ReflectionPad.test_3d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_reflection_pad_2d(request, kwargs):
    ReflectionPad.test_4d(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_relu(request, kwargs):
    Relu.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_relu6(request, kwargs):
    Relu6.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_remainder(request, kwargs):
    Remainder.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_repeat(request, kwargs):
    Repeat.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_reshape_2d_to_4d_random_reshape(request, kwargs):
    Reshape.test_2d_to_4d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_reshape_2d_to_4d_flatten_last_two_dims(request, kwargs):
    Reshape.test_2d_to_4d_flatten_last_two_dims(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_reshape_5d_random_reshape(request, kwargs):
    Reshape.test_5d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_reshape_5d_flatten_last_two_dims(request, kwargs):
    Reshape.test_5d_flatten_last_two_dims(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_rms_norm(request, kwargs):
    RmsNorm.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_roll(request, kwargs):
    Roll.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_round(request, kwargs):
    Round.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_rsqrt(request, kwargs):
    Rsqrt.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        Tolerance(),
        pytest.raises(AssertionError, match=EXPECT_NOT_FULLY_DELEGATED),
        Tolerance(rtol=1e-1),
    ]
)
@with_htp_context
def test_sdpa(request, kwargs):
    ScaledDotProductAttention.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_scatter_src(request, kwargs):
    ScatterSrc.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_select_copy(request, kwargs):
    SelectCopy.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_select_scatter(request, kwargs):
    SelectScatter.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_sigmoid(request, kwargs):
    Sigmoid.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_sign(request, kwargs):
    Sign.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_sin(request, kwargs):
    Sin.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_slice_copy(request, kwargs):
    SliceCopy.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_slice_scatter(request, kwargs):
    SliceScatter.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_softmax(request, kwargs):
    Softmax.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_split(request, kwargs):
    Split.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_square(request, kwargs):
    Square.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_squeeze(request, kwargs):
    Squeeze.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_stack(request, kwargs):
    Stack.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_sum_int_list(request, kwargs):
    SumIntList.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_swapaxes(request, kwargs):
    SwapAxes.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_tan(request, kwargs):
    Tan.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_tanh(request, kwargs):
    Tanh.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_threshold(request, kwargs):
    Threshold.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_triu(request, kwargs):
    Triu.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_triu_constant(request, kwargs):
    Triu.test_constant(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_trunc(request, kwargs):
    Trunc.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_topk(request, kwargs):
    TopK.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_unbind(request, kwargs):
    Unbind.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_unflatten(request, kwargs):
    Unflatten.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_unfold(request, kwargs):
    Unfold.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype(
    [
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
        pytest.raises(Exception, check=check_exception(EXCEPTION_FROM_PASSES)),
    ]
)
@with_htp_context
def test_unfold_unsupported(request, kwargs):
    Unfold.test_unsupported(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_unsqueeze(request, kwargs):
    Unsqueeze.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_view_2d_to_4d_random_reshape(request, kwargs):
    View.test_2d_to_4d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_view_2d_to_4d_flatten_last_two_dims(request, kwargs):
    View.test_2d_to_4d_flatten_last_two_dims(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_view_5d_random_reshape(request, kwargs):
    View.test_5d_random_reshape(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_view_5d_flatten_last_two_dims(request, kwargs):
    View.test_5d_flatten_last_two_dims(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_where(request, kwargs):
    Where.test(request, kwargs)  # noqa: F405


@enumerate_activation_dtype([Tolerance(), Tolerance(), Tolerance(rtol=1e-1)])
@with_htp_context
def test_var(request, kwargs):
    Var.test(request, kwargs)  # noqa: F405
