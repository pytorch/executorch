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
 * Packed INT6 matrix multiplication for GGUF Q6_K weights (symmetric).
 *
 * The 6-bit weight is split into two planes plus a per-group scale; there is
 * NO zero tensor — Q6_K is symmetric and the stored value is u = q + 32 in
 * [0, 63] (q in [-32, 31]), with the constant -32 offset applied in the kernel.
 *
 * Weight format:
 *   ql    : [N, K/2] uint8 — low-nibble plane, nibble-packed even/odd
 *           (ql[:,j] = (u[:,2j] & 0xF) | ((u[:,2j+1] & 0xF) << 4)).
 *   qh    : [N, K/4] uint8 — high-2-bit plane, 4 values/byte, arranged per
 *           32-weight chunk as hi_even_packed[4] then hi_odd_packed[4]; each
 *           byte holds the four 2-bit highs of one dp4a word, bit field j
 *           (bits 2j..2j+1) = high 2 bits of that word's j-th even/odd weight.
 *   scale : [N, K//group_size] int8 per-group scale codes (row-major),
 *           with a per-row [N, 1] bf16 super-scale ``steps`` so the
 *           group scale is decoded as code * step (7.0 -> 6.5 bpw).
 * W6A8 dp4a matvec: dynamically quantizes activations to INT8, reconstructs
 * full 6-bit weight bytes, then uses dp4a for fused int6xint8 dot products.
 *
 * @param self     Input activation [M, K] bf16
 * @param ql       Low-nibble plane [N, K/2] uint8
 * @param qh       High-2-bit plane [N, K/4] uint8
 * @param scale    Per-group scale codes [N, K//group_size] int8
 * @param steps    Per-row super-scale [N, 1] bf16 (scale = code * step)
 * @param group_size Quantization group size (multiple of 8; e.g. 16 for Q6_K)
 * @param ret0     Output [M, N] bf16
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_int6_plain_mm(
    Tensor* self,
    Tensor* ql,
    Tensor* qh,
    Tensor* scale,
    Tensor* steps,
    int64_t group_size,
    Tensor** ret0);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
