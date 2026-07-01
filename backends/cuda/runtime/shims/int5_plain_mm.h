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
 * Packed INT5 matrix multiplication for GGUF Q5_K weights (asymmetric).
 *
 * The 5-bit weight is split into two planes plus a per-group scale AND zero
 * point — Q5_K is asymmetric (it has both ``d`` and ``dmin``), so the stored
 * value is the raw unsigned u = q in [0, 31] and a per-group zero point is
 * subtracted in the kernel (like INT4, unlike the symmetric INT6 path).
 *
 * Weight format:
 *   ql    : [N, K/2] uint8 — low-nibble plane, nibble-packed even/odd
 *           (ql[:,j] = (u[:,2j] & 0xF) | ((u[:,2j+1] & 0xF) << 4)).
 *   qh    : [N, K/8] uint8 — high-1-bit plane, 8 values/byte, arranged per
 *           32-weight chunk as 4 bytes (one per dp4a word); each byte holds the
 *           four 1-bit highs of that word's even weights in the low nibble and
 *           its odd weights in the high nibble.
 *   scale : [N, K//group_size] uint8 per-group scale codes (row-major).
 *   zero  : [N, K//group_size] uint8 per-group zero codes (row-major).
 *           with a per-row [N, 2] bf16 super-scale ``steps`` so the group
 *           scale/zero are decoded as code*scale_step / code*zero_step
 *           (5.625 -> 5.125 bpw).
 * W5A8 dp4a matvec: dynamically quantizes activations to INT8, reconstructs
 * full 5-bit weight bytes, then uses dp4a for fused int5×int8 dot products.
 *
 * @param self     Input activation [M, K] bf16
 * @param ql       Low-nibble plane [N, K/2] uint8
 * @param qh       High-1-bit plane [N, K/8] uint8
 * @param scale    Per-group scale codes [N, K//group_size] uint8
 * @param zero     Per-group zero codes [N, K//group_size] uint8
 * @param steps    Per-row super-scale [N, 2] bf16 (scale_step, zero_step)
 * @param group_size Quantization group size (multiple of 32; e.g. 32 for Q5_K)
 * @param ret0     Output [M, N] bf16
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_int5_plain_mm(
    Tensor* self,
    Tensor* ql,
    Tensor* qh,
    Tensor* scale,
    Tensor* zero,
    Tensor* steps,
    int64_t group_size,
    Tensor** ret0);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
