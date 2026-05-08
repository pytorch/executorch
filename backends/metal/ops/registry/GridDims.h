/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Kernel-launch dimension helpers — pick threadgroup shapes and grid sizes
// that match the work and stay under hardware limits.

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <climits>
#include <cstdint>
#include <tuple>
#include <utility>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Pick a power-of-two threadgroup shape (block_x, block_y, block_z) that
// fits dim0/dim1/dim2 and whose total thread count is at most 2^maxPow2.
// Mirrors MLX's get_block_dims_common. Default maxPow2=10 (cap = 1024
// threads/threadgroup, the Apple-Silicon hardware limit).
inline std::tuple<uint32_t, uint32_t, uint32_t>
getBlockDims(int dim0, int dim1, int dim2, int maxPow2 = 10) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    if (dim0 >= (1 << (pows[0] + 1))) { pows[0]++; sum++; }
    if (sum == maxPow2) break;
    if (dim1 >= (1 << (pows[1] + 1))) { pows[1]++; sum++; }
    if (sum == maxPow2) break;
    if (dim2 >= (1 << (pows[2] + 1))) { pows[2]++; sum++; }
    if (sum == presum || sum == maxPow2) break;
  }
  return std::make_tuple<uint32_t, uint32_t, uint32_t>(
      1u << pows[0], 1u << pows[1], 1u << pows[2]);
}

// Factor a flat element count into a 2-D grid (gx, gy) where each axis fits
// in uint32_t. Use this when numel > UINT32_MAX would overflow a 1-D grid.
// Mirrors MLX's get_2d_grid_dims_common (without strides).
inline std::pair<uint32_t, uint32_t> get2DGridDims(
    uint64_t numel, uint64_t workPerThread = 1) {
  uint64_t threads = (numel + workPerThread - 1) / workPerThread;
  if (threads == 0) return {1u, 1u};
  if (threads <= UINT32_MAX) return {static_cast<uint32_t>(threads), 1u};
  uint64_t gy = (threads + UINT32_MAX - 1) / UINT32_MAX;
  uint64_t gx = (threads + gy - 1) / gy;
  if (gx > UINT32_MAX || gy > UINT32_MAX) {
    gx = UINT32_MAX;
    gy = UINT32_MAX;
  }
  return {static_cast<uint32_t>(gx), static_cast<uint32_t>(gy)};
}

// Recommended elements per thread for elementwise kernels, by dtype size.
// Smaller dtypes -> more elements per thread (better bandwidth + larger
// vectorized loads). Mirrors MLX's WorkPerThread<T> trait.
inline int workPerThread(exec_aten::ScalarType dtype) {
  using ScalarType = exec_aten::ScalarType;
  switch (dtype) {
    case ScalarType::Bool:
    case ScalarType::Byte:
    case ScalarType::Char:  return 8;  // 1-byte: 8 elems = 8 bytes
    case ScalarType::Short:
    case ScalarType::Half:  return 8;  // 2-byte: 8 elems = 16 bytes
    case ScalarType::Int:
    case ScalarType::Float: return 4;  // 4-byte: 4 elems = 16 bytes
    case ScalarType::Long:
    case ScalarType::Double: return 2; // 8-byte: 2 elems = 16 bytes
    default: return 4;
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
