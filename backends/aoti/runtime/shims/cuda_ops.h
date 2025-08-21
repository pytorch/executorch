/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include "tensor_attribute.h"

namespace executorch {
namespace backends {
namespace aoti {

using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

extern "C" {

// CUDA addmm operation: out = beta * self + alpha * (mat1 @ mat2)
AOTITorchError aoti_torch_cuda_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha);

// CUDA convolution operation
AOTITorchError aoti_torch_cuda_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle* bias,
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
    AtenTensorHandle* ret0);

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch