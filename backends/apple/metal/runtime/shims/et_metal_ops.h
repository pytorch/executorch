/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/types.h>

namespace executorch {
namespace backends {
namespace metal {

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ExecuTorch implementation of aoti_torch_mps_mm_out.
 * Performs simple matrix multiplication: out = self @ mat2
 */
AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2);

/**
 * ExecuTorch implementation of aoti_torch_mps_bmm_out.
 * Performs batched matrix multiplication: out = self @ mat2
 * All tensors must be 3-D with matching batch dimensions.
 */
AOTITorchError aoti_torch_mps_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2);

/**
 * ExecuTorch implementation of aoti_torch_mps_convolution.
 * Performs 2D convolution operation - matches PyTorch AOTI signature
 */
AOTITorchError aoti_torch_mps_convolution(
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle* bias,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len_,
    int64_t groups,
    AOTITensorHandle* ret0);

/**
 * ExecuTorch implementation of
 * aoti_torch_mps__scaled_dot_product_attention_math_for_mps. Performs scaled
 * dot product attention calculation - matches PyTorch AOTI signature
 */
AOTITorchError aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
    AOTITensorHandle query,
    AOTITensorHandle key,
    AOTITensorHandle value,
    AOTITensorHandle* attn_mask,
    double dropout_p,
    int32_t is_causal,
    AOTITensorHandle* dropout_mask,
    double* scale,
    AOTITensorHandle* ret0,
    AOTITensorHandle* ret1);

AOTITorchError aoti_torch_mps__linear_fp_act_4bit_weight(
    AOTITensorHandle A,
    AOTITensorHandle B,
    int64_t group_size,
    AOTITensorHandle S,
    AOTITensorHandle Z,
    AOTITensorHandle* ret);

#ifdef __cplusplus
} // extern "C"
#endif

} // namespace metal
} // namespace backends
} // namespace executorch
