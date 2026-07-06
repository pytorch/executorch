/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/backends/aoti/export.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/core/slim_tensor_view_incl.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {

using executorch::runtime::Error;
using AOTITorchError = Error;

using SlimTensor = executorch::backends::aoti::slim::SlimTensor;

extern "C" {

/**
 * Generates a tensor filled with uniform random values in [0, 1), matching
 * the behavior of torch.rand / aten::rand (see
 * aten/src/ATen/native/cuda/DistributionUniform.cu and the
 * `transformation::uniform_real` helper in
 * aten/src/ATen/native/cuda/DistributionTemplates.h).
 *
 * Implements the AOTI shim for aten::rand.default on CUDA. Uses cuRAND
 * Philox counter-based RNG with GPU-resident state, then maps the random
 * uint32 to [0, 1) using PyTorch's bit-mask + divisor formulation rather
 * than curand_uniform (which returns (0, 1]). The counter is atomically
 * advanced by each kernel invocation on-device, making it fully compatible
 * with CUDA graph capture and replay — each replay produces different
 * values because the counter increments on the GPU.
 *
 * Supports float32 and bfloat16 output dtypes.
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_rand(
    const int64_t* size,
    int64_t size_len_,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t device_index_,
    int32_t* pin_memory,
    SlimTensor** ret0);

/**
 * Fills a pre-allocated int64 tensor with random integers in [low, high),
 * matching the behavior of torch.randint / aten::randint.low_out (see
 * `transformation::uniform_int_from_to` in
 * aten/src/ATen/native/cuda/DistributionTemplates.h).
 *
 * Implements the AOTI shim for aten::randint.low_out on CUDA. Used by
 * Inductor's Philox RNG to generate random seeds. Each thread atomically
 * advances a GPU-resident counter for unique offsets, making this fully
 * compatible with CUDA graph capture and replay.
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_randint_low_out(
    SlimTensor* out,
    int64_t low,
    int64_t high,
    const int64_t* size,
    int64_t size_len_);

} // extern "C"

} // namespace executorch::backends::cuda
