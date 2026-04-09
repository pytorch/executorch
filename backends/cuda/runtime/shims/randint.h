/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using SlimTensor = executorch::backends::aoti::slim::SlimTensor;

extern "C" {

/**
 * Fills a pre-allocated CUDA tensor with random integers in [low, high).
 *
 * Used by AOTI-generated code when the model calls torch.randint or ops
 * that decompose into randint (e.g. torch.rand_like on some dtypes).
 *
 * @param out Pre-allocated output tensor on CUDA (must not be null).
 * @param low Lower bound (inclusive) of the random range.
 * @param high Upper bound (exclusive) of the random range.
 * @param size Pointer to array of output dimension sizes.
 * @param size_len_ Number of dimensions.
 * @return AOTITorchError error code (Error::Ok on success).
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_randint_low_out(
    SlimTensor* out,
    int64_t low,
    int64_t high,
    const int64_t* size,
    int64_t size_len_);

} // extern "C"

} // namespace executorch::backends::cuda
