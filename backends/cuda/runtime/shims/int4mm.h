/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Performs quantized INT4 matrix multiplication.
 *
 * INT4 weights are stored in a packed tensor core layout optimized for
 * NVIDIA Ampere+ GPUs (sm_80+) using m16n8k16 tensor core tiles.
 *
 * HARDWARE REQUIREMENTS:
 * - CUDA Compute Capability >= 8.0 (Ampere or later)
 * - BFloat16 support (native on sm_80+)
 *
 * TENSOR REQUIREMENTS:
 * @param self Input activation matrix [m, k]
 *   - Must be BFloat16 dtype
 *   - Must be 2D
 *   - Must be on CUDA device
 *   - Row-major layout (contiguous)
 *
 * @param mat2 Quantized weight matrix in packed tensor core layout
 *   - Must be Int32 dtype (contains packed INT4 values)
 *   - Must be 4D: [n/8][k/(InnerKTiles*16)][32][InnerKTiles/2]
 *     where InnerKTiles = 2, 4, or 8
 *   - Each Int32 contains 8 packed INT4 values
 *   - Layout optimized for tensor core access patterns
 *   - Must be on CUDA device
 *
 * @param qGroupSize Quantization group size (number of values sharing
 * scale/zero)
 *   - Must be one of: 32, 64, 128, or 256
 *   - Smaller groups = higher accuracy but more metadata
 *   - Must evenly divide k dimension
 *
 * @param qScaleAndZeros Dequantization parameters [k/qGroupSize][n][2]
 *   - Must be BFloat16 dtype
 *   - Must be 3D
 *   - [:, :, 0] contains scales
 *   - [:, :, 1] contains zero points
 *   - Must be on CUDA device
 *
 * @param ret0 Output parameter for result matrix [m, n]
 *   - Allocated by this function as BFloat16
 *   - Must not be null
 *   - Caller is responsible for freeing via aoti_torch_delete_tensor_object()
 *
 * @return AOTITorchError error code:
 *   - Error::Ok: Success
 *   - Error::InvalidArgument: Null pointer, wrong dtype, wrong dimensions,
 *     or invalid qGroupSize
 *   - Error::Internal: CUDA kernel launch failure
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda__weight_int4pack_mm(
    Tensor* self,
    Tensor* mat2,
    int64_t qGroupSize,
    Tensor* qScaleAndZeros,
    Tensor** ret0);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
