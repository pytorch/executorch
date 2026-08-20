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
 * INT8 matrix multiplication reading plain (unpacked) int8 weights.
 *
 * Weight format: [N, K] int8, one value per element (natural k order).
 * Scale: [N, K//group_size] bf16 per-group scales
 *        (IntxUnpackedToInt8Tensor layout, row-major).
 * Zero:  [N, K//group_size] int8 per-group zero points.
 * W8A8 dp4a matvec: dynamically quantizes activations to INT8,
 * then uses dp4a for fused int8×int8 dot products.
 *
 * @param self     Input activation [M, K] bf16
 * @param qdata    Weights [N, K] int8
 * @param scale    Per-group scales [N, K//group_size] bf16
 * @param zero     Per-group zero points [N, K//group_size] int8
 * @param group_size Quantization group size (multiple of 32)
 * @param ret0     Output [M, N] bf16
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_int8_plain_mm(
    Tensor* self,
    Tensor* qdata,
    Tensor* scale,
    Tensor* zero,
    int64_t group_size,
    Tensor** ret0);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
