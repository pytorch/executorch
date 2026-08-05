# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)

from executorch.backends.qualcomm.tests.rework.passes.conftest import (
    enumerate_backends,
    enumerate_backends_quantized,
    repack_pass_fixtures,
)
from executorch.backends.qualcomm.tests.rework.src.pattern import *  # noqa: F403


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_annotate_avg_pool1d(request, kwargs):
    AnnotateAvgPool1D.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_annotate_quant_attrs(request, kwargs):
    AnnotateQuantAttrs.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_annotate_stack(request, kwargs):
    AnnotateStack.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_annotate_unbind(request, kwargs):
    AnnotateUnbind.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_build_quant_io(request, kwargs):
    BuildQuantIo.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_canonicalize_conv(request, kwargs):
    CanonicalizeConv.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_convert_bmm_to_matmul(request, kwargs):
    ConvertBmmToMatmul.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_convert_linear_to_conv2d(request, kwargs):
    ConvertLinearToConv2d.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_convert_mha_to_sha(request, kwargs):
    ConvertMhaToSha.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_convert_square_to_pow(request, kwargs):
    ConvertSquareToPow.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_acos(request, kwargs):
    DecomposeAcos.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_addmm(request, kwargs):
    DecomposeAddmm.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_any(request, kwargs):
    DecomposeAny.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_atan2(request, kwargs):
    DecomposeAtan2.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_binary_alpha(request, kwargs):
    DecomposeBinaryAlpha.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_cdist(request, kwargs):
    DecomposeCDist.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_col_im(request, kwargs):
    DecomposeColIm.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_diagonal(request, kwargs):
    DecomposeDiagonal.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_div_mode(request, kwargs):
    DecomposeDivMode.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_einsum(request, kwargs):
    DecomposeEinsum.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_expm1(request, kwargs):
    DecomposeExpM1.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_fill(request, kwargs):
    DecomposeFill.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_floor_divide(request, kwargs):
    DecomposeFloorDivide.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_glu(request, kwargs):
    DecomposeGlu.test(request, kwargs)  # noqa: F405


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"backend_type": QnnExecuTorchBackendType.kLpaiBackend},
            id=str(QnnExecuTorchBackendType.kLpaiBackend),
        )
    ],
)
@repack_pass_fixtures
def test_decompose_hardsigmoid(request, kwargs):
    DecomposeHardsigmoid.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_hyperbolic_variants(request, kwargs):
    DecomposeHyperbolicVariants.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_linalg_vector_norm(request, kwargs):
    DecomposeLinalgVectorNorm.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_log_variants(request, kwargs):
    DecomposeLogVariants.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_max_pool3d(request, kwargs):
    DecomposeMaxPool3d.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_min_max_dim(request, kwargs):
    DecomposeMinMaxDim.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_pad(request, kwargs):
    DecomposePad.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_reciprocal(request, kwargs):
    DecomposeReciprocal.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_remainder(request, kwargs):
    DecomposeRemainder.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_roll(request, kwargs):
    DecomposeRoll.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_select_scatter(request, kwargs):
    DecomposeSelectScatter.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_silu(request, kwargs):
    DecomposeSilu.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_tan(request, kwargs):
    DecomposeTan.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_threshold(request, kwargs):
    DecomposeThreshold.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_triu(request, kwargs):
    DecomposeTriu.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_trunc(request, kwargs):
    DecomposeTrunc.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_var(request, kwargs):
    DecomposeVar.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_decompose_wrap_with_autocast(request, kwargs):
    DecomposeWrapWithAutocast.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_i64_to_i32(request, kwargs):
    I64toI32.test(request, kwargs)  # noqa: F405


@enumerate_backends([QnnExecuTorchBackendType.kHtpBackend])
@repack_pass_fixtures
def test_insert_io_qdq(request, kwargs):
    InsertIOQDQ.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_expand_broadcast_tensor_shape(request, kwargs):
    ExpandBroadcastTensorShape.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_fixed_linear_keep_dim(request, kwargs):
    FixedLinearKeepDim.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_fold_qdq(request, kwargs):
    FoldQDQ.test(request, kwargs)  # noqa: F405


# Test case tests pytorch dtype conversion,
# since LPAI only supports quant type casting, skip LPAI for now.
@enumerate_backends(
    [
        QnnExecuTorchBackendType.kGpuBackend,
        QnnExecuTorchBackendType.kHtpBackend,
    ]
)
@repack_pass_fixtures
def test_fuse_consecutive_cast(request, kwargs):
    FuseConsecutiveCast.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_fuse_consecutive_transpose(request, kwargs):
    FuseConsecutiveTranspose.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_insert_requantize(request, kwargs):
    InsertRequantize.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_insert_reshape_for_reduce_ops(request, kwargs):
    InsertReshapeForReduceOps.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_layout_transform(request, kwargs):
    LayoutTransform.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_lift_constant_scalar_operands(request, kwargs):
    LiftConstantScalarOperands.test(request, kwargs)  # noqa: F405


@enumerate_backends([QnnExecuTorchBackendType.kLpaiBackend])
@repack_pass_fixtures
def test_lpai_partition_fallback_support(request, kwargs):
    LpaiPartitionFallbackSupport.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_recompose_pad_maxpool2d(request, kwargs):
    RecomposePadMaxPool2d.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_recompose_pixel_unshuffle(request, kwargs):
    RecomposePixelUnshuffle.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_recompose_rms_norm(request, kwargs):
    RecomposeRmsNorm.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_remove_0d_tensor(request, kwargs):
    Remove0DTensor.test(request, kwargs)  # noqa: F405


@enumerate_backends()
@repack_pass_fixtures
def test_remove_redundancy(request, kwargs):
    RemoveRedundancy.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_replace_arange_args(request, kwargs):
    ReplaceArangeArgs.test(request, kwargs)  # noqa: F405


@enumerate_backends([QnnExecuTorchBackendType.kHtpBackend])
@repack_pass_fixtures
def test_resolve_debug_handle(request, kwargs):
    ResolveDebugHandle.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_seq_mse(request, kwargs):
    SeqMSE.test(request, kwargs)  # noqa: F405


@enumerate_backends_quantized()
@repack_pass_fixtures
def test_tag_quant_io(request, kwargs):
    TagQuantIO.test(request, kwargs)  # noqa: F405
