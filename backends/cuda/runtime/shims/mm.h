/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Matrix multiplication via cuBLAS: out = self @ mat2.
 *
 * Replaces libtorch's aoti_torch_cuda_mm_out so the AOTI CUDA backend
 * can run without libtorch_cuda.so. Calls cublasGemmEx directly.
 *
 * @param out  Pre-allocated output [M, N], same dtype as inputs.
 * @param self Input matrix [M, K]. Must be bf16 or fp16, 2D, contiguous.
 * @param mat2 Input matrix [K, N]. Must be bf16 or fp16, 2D, contiguous.
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda_mm_out(Tensor* out, Tensor* self, Tensor* mat2);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
