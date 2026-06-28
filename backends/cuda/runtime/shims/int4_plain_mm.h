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
 * INT4 matrix multiplication reading plain nibble-packed weights.
 *
 * Weight format: [N, K//2] uint8, two INT4 values per byte
 * (low nibble = even k, high nibble = odd k).
 * Scale: [N, K//group_size] uint8 per-group scale codes (coalesced layout).
 * Zero:  [N, K//group_size] uint8 per-group zero codes (coalesced layout).
 * Steps: [N, 2] bf16 per-row super-scales (scale_step, zero_step); the group
 *        scale/zero are decoded as code*scale_step / code*zero_step. This
 *        halves the per-group metadata vs bf16 scale+zero (5.0 -> 4.5 bpw).
 * W4A8 dp4a matvec: dynamically quantizes activations to INT8,
 * then uses dp4a for fused int4×int8 dot products.
 *
 * @param self     Input activation [M, K] bf16
 * @param qdata    Packed weights [N, K//2] uint8
 * @param scale    Per-group scale codes [N, K//group_size] uint8
 * @param zero     Per-group zero codes [N, K//group_size] uint8
 * @param steps    Per-row super-scales [N, 2] bf16 (scale_step, zero_step)
 * @param group_size Quantization group size (32, 64, 128)
 * @param ret0     Output [M, N] bf16
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_int4_plain_mm(
    Tensor* self,
    Tensor* qdata,
    Tensor* scale,
    Tensor* zero,
    Tensor* steps,
    int64_t group_size,
    Tensor** ret0);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
