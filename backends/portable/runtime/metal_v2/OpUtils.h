/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Shared host-side helpers for MetalOps.
//
// Inspired by mlx/backend/common/binary.h and mlx/backend/common/utils.h.
// All functions are pure, header-only, and do not allocate GPU resources --
// they only inspect Tensor metadata (sizes/strides) and produce small host
// buffers that ops can pass to their kernels.
//
// Naming convention mirrors the equivalent shader-side helpers in
// metal_v2/kernels/accessors.metal.h (to be added separately).

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::etensor::Tensor;
using exec_aten::ArrayRef;
using exec_aten::SizesType;

//===----------------------------------------------------------------------===//
// ElementwiseVariant
//
// Classifies an elementwise binary op's input layout to pick the fastest
// kernel specialization. Mirrors mlx::core::BinaryOpType.
//===----------------------------------------------------------------------===//

enum class ElementwiseVariant {
  ScalarScalar,  // both inputs are 1-element
  ScalarVector,  // a is scalar, b is contiguous vector
  VectorScalar,  // a is contiguous vector, b is scalar
  VectorVector,  // both inputs are same shape and contiguous
  General,       // arbitrary strides / broadcast required
};

inline const char* variantPrefix(ElementwiseVariant v) {
  switch (v) {
    case ElementwiseVariant::ScalarScalar: return "ss";
    case ElementwiseVariant::ScalarVector: return "sv";
    case ElementwiseVariant::VectorScalar: return "vs";
    case ElementwiseVariant::VectorVector: return "vv";
    case ElementwiseVariant::General:      return "g";
  }
  return "g";
}

// Returns true if all dims of `t` have stride matching a packed row-major
// (innermost-fastest) layout. Equivalent to MLX's `flags().row_contiguous`.
inline bool isRowContiguous(const Tensor& t) {
  auto sizes = t.sizes();
  auto strides = t.strides();
  if (sizes.size() != strides.size()) return false;
  if (sizes.empty()) return true;
  int64_t expected = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    if (sizes[i] == 1) continue;          // size-1 dim's stride is irrelevant
    if (strides[i] != expected) return false;
    expected *= sizes[i];
  }
  return true;
}

inline bool sameShape(const Tensor& a, const Tensor& b) {
  auto as = a.sizes();
  auto bs = b.sizes();
  if (as.size() != bs.size()) return false;
  for (size_t i = 0; i < as.size(); ++i) {
    if (as[i] != bs[i]) return false;
  }
  return true;
}

inline ElementwiseVariant classifyBinary(const Tensor& a, const Tensor& b) {
  bool a_scalar = (a.numel() == 1);
  bool b_scalar = (b.numel() == 1);
  if (a_scalar && b_scalar) return ElementwiseVariant::ScalarScalar;
  if (a_scalar && isRowContiguous(b)) return ElementwiseVariant::ScalarVector;
  if (b_scalar && isRowContiguous(a)) return ElementwiseVariant::VectorScalar;
  if (sameShape(a, b) && isRowContiguous(a) && isRowContiguous(b)) {
    return ElementwiseVariant::VectorVector;
  }
  return ElementwiseVariant::General;
}

//===----------------------------------------------------------------------===//
// broadcastStrides
//
// Compute strides for `shape` aligned to `out_shape` such that broadcast
// dimensions (where the input has size 1 or is missing) get stride 0.
// Used to pass per-input stride arrays to General kernels.
//
// Example: in_shape = [3, 1, 5], out_shape = [2, 3, 4, 5]
//          -> strides = [0 (broadcast), 5 (was 3), 0 (was 1), 1]
//===----------------------------------------------------------------------===//

inline std::vector<int64_t> broadcastStrides(
    ArrayRef<SizesType> in_shape,
    const std::vector<SizesType>& out_shape) {
  std::vector<int64_t> strides(out_shape.size(), 0);
  int offset = static_cast<int>(out_shape.size()) -
               static_cast<int>(in_shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(in_shape.size()) - 1; i >= 0; --i) {
    if (in_shape[i] == out_shape[i + offset]) {
      strides[i + offset] = stride;
      stride *= in_shape[i];
    }
    // else: input dim is 1 (broadcast) or missing -> stride stays 0
  }
  return strides;
}

//===----------------------------------------------------------------------===//
// collapseContiguousDims
//
// Merges adjacent dims that are contiguous in *every* input's stride layout.
// Reduces the effective ndim passed to the General kernel, which lowers
// per-element index arithmetic and lets more cases hit the fast paths.
//
// Returns:
//   .first  = collapsed shape (length <= original ndim)
//   .second = vector of collapsed strides, one per input (same length as .first)
//
// For a single contiguous tensor this collapses everything to a 1-D shape.
// For broadcast (one input has stride 0 along a dim) the dim is preserved.
//
// Mirrors mlx::core::collapse_contiguous_dims for the multi-array case.
//===----------------------------------------------------------------------===//

inline std::pair<std::vector<SizesType>, std::vector<std::vector<int64_t>>>
collapseContiguousDims(
    const std::vector<SizesType>& shape,
    const std::vector<std::vector<int64_t>>& strides_per_input) {
  const size_t ndim = shape.size();
  const size_t nin = strides_per_input.size();

  std::vector<SizesType> out_shape;
  std::vector<std::vector<int64_t>> out_strides(nin);

  if (ndim == 0) {
    return {out_shape, out_strides};
  }

  // Start with the innermost dim.
  out_shape.push_back(shape[ndim - 1]);
  for (size_t k = 0; k < nin; ++k) {
    out_strides[k].push_back(strides_per_input[k][ndim - 1]);
  }

  // Walk outward; merge dim i into the current group if EVERY input's
  // stride[i] equals stride[i+1] * shape[i+1] (i.e. truly adjacent in memory)
  // OR all inputs have stride 0 along both dims (still safe to merge).
  for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
    SizesType inner_size = out_shape.back();
    bool can_merge = (shape[i] == 1) || (inner_size == 1);
    if (!can_merge) {
      can_merge = true;
      for (size_t k = 0; k < nin; ++k) {
        int64_t outer = strides_per_input[k][i];
        int64_t inner = out_strides[k].back();
        bool both_zero = (outer == 0) && (inner == 0);
        bool packed = (outer == inner * inner_size);
        if (!both_zero && !packed) {
          can_merge = false;
          break;
        }
      }
    }
    if (can_merge) {
      out_shape.back() = shape[i] * inner_size;
      // strides for the merged group take the inner stride
      for (size_t k = 0; k < nin; ++k) {
        if (out_strides[k].back() == 0 && strides_per_input[k][i] != 0) {
          out_strides[k].back() = strides_per_input[k][i];
        }
        // else keep inner stride
      }
    } else {
      out_shape.push_back(shape[i]);
      for (size_t k = 0; k < nin; ++k) {
        out_strides[k].push_back(strides_per_input[k][i]);
      }
    }
  }

  // We built the lists innermost-first; reverse to outermost-first.
  std::reverse(out_shape.begin(), out_shape.end());
  for (auto& s : out_strides) {
    std::reverse(s.begin(), s.end());
  }
  return {out_shape, out_strides};
}

//===----------------------------------------------------------------------===//
// makeContiguousStrides
//
// Build packed row-major strides for `shape`.
// e.g. shape [2, 3, 4] -> strides [12, 4, 1]
//===----------------------------------------------------------------------===//

inline std::vector<int64_t> makeContiguousStrides(
    const std::vector<SizesType>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * static_cast<int64_t>(shape[i]);
  }
  return strides;
}

inline std::vector<int64_t> makeContiguousStrides(ArrayRef<SizesType> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * static_cast<int64_t>(shape[i]);
  }
  return strides;
}

//===----------------------------------------------------------------------===//
// isColContiguous
//
// True if `t` is column-major contiguous (innermost dim has the largest
// stride). Mirror of isRowContiguous.
//===----------------------------------------------------------------------===//

inline bool isColContiguous(const Tensor& t) {
  auto sizes = t.sizes();
  auto strides = t.strides();
  if (sizes.size() != strides.size()) return false;
  if (sizes.empty()) return true;
  int64_t expected = 1;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 1) continue;
    if (strides[i] != expected) return false;
    expected *= sizes[i];
  }
  return true;
}

//===----------------------------------------------------------------------===//
// getBlockDims
//
// Pick a power-of-two threadgroup shape (block_x, block_y, block_z) that
// fits dim0/dim1/dim2 and whose total thread count is at most 2^maxPow2.
// Mirrors MLX's get_block_dims_common.
//
// Default maxPow2 = 10 (cap = 1024 threads/threadgroup, the Apple-Silicon
// hardware limit).
//===----------------------------------------------------------------------===//

inline std::tuple<uint32_t, uint32_t, uint32_t> getBlockDims(
    int dim0, int dim1, int dim2, int maxPow2 = 10) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    if (dim0 >= (1 << (pows[0] + 1)))      { pows[0]++; sum++; }
    if (sum == maxPow2) break;
    if (dim1 >= (1 << (pows[1] + 1)))      { pows[1]++; sum++; }
    if (sum == maxPow2) break;
    if (dim2 >= (1 << (pows[2] + 1)))      { pows[2]++; sum++; }
    if (sum == presum || sum == maxPow2) break;
  }
  return std::make_tuple<uint32_t, uint32_t, uint32_t>(
      1u << pows[0], 1u << pows[1], 1u << pows[2]);
}

//===----------------------------------------------------------------------===//
// get2DGridDims
//
// Factor a flat element count into a 2-D grid (gx, gy) where each axis fits
// in uint32_t. Use this when `numel > UINT32_MAX` would overflow a 1-D grid.
// Returned values multiplied together cover at least `numel / workPerThread`
// threads (so divide your total work by `workPerThread` if each thread
// processes more than one element).
//
// Mirrors MLX's get_2d_grid_dims_common (without strides — we handle only
// the simple "flat numel" case here; broadcast/stride-aware factoring can be\n// added when needed).
//===----------------------------------------------------------------------===//

inline std::pair<uint32_t, uint32_t> get2DGridDims(
    uint64_t numel, uint64_t workPerThread = 1) {
  uint64_t threads = (numel + workPerThread - 1) / workPerThread;
  if (threads == 0) {
    return {1u, 1u};
  }
  if (threads <= UINT32_MAX) {
    return {static_cast<uint32_t>(threads), 1u};
  }
  // Find the smallest gy such that ceil(threads / gy) fits in uint32_t.
  uint64_t gy = (threads + UINT32_MAX - 1) / UINT32_MAX;
  uint64_t gx = (threads + gy - 1) / gy;
  if (gx > UINT32_MAX || gy > UINT32_MAX) {
    // Caller must have an absurdly large tensor (>2^64 elements). Clamp.
    gx = UINT32_MAX;
    gy = UINT32_MAX;
  }
  return {static_cast<uint32_t>(gx), static_cast<uint32_t>(gy)};
}

//===----------------------------------------------------------------------===//
// workPerThread
//
// Returns the recommended number of elements each thread should process for
// elementwise kernels, based on dtype size. Smaller dtypes -> more elements
// per thread (better memory bandwidth utilization, larger vectorized loads).
// Mirrors mlx's WorkPerThread<T> trait.
//
// Use as the `N` template parameter for the binary_v* / unary_v* kernels.
//===----------------------------------------------------------------------===//

inline int workPerThread(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Bool:
    case ScalarType::Byte:
    case ScalarType::Char:
      return 8;   // 1-byte: 8 elems = 8 bytes (one i64 load)
    case ScalarType::Short:
      return 8;   // 2-byte: 8 elems = 16 bytes (one float4-equivalent)
    case ScalarType::Half:
      return 8;   // 2-byte half: same
    case ScalarType::Int:
    case ScalarType::Float:
      return 4;   // 4-byte: 4 elems = 16 bytes (one float4)
    case ScalarType::Long:
    case ScalarType::Double:
      return 2;   // 8-byte: 2 elems = 16 bytes
    default:
      return 4;
  }
}

//===----------------------------------------------------------------------===//
// DeviceTier + getDeviceTier
//
// Coarse classification of the GPU's perf bucket. Used by ops that want
// to pick different tile sizes / thresholds per device class. Mirrors v1's
// MatMulConfig::forDevice but lifted to a generic helper.
//
// Caller passes the device name (e.g. [[device name] UTF8String]) to keep
// this header free of Metal imports.
//===----------------------------------------------------------------------===//

enum class DeviceTier {
  Phone,     // iPhone, iPad
  MacBase,   // M-series base / Pro
  MacUltra,  // M-series Max / Ultra
};

inline DeviceTier getDeviceTierFromName(const char* deviceName) {
  if (deviceName == nullptr) return DeviceTier::MacBase;
  // Order matters: check Ultra/Max before "M" prefix matches.
  if (std::strstr(deviceName, "Ultra") || std::strstr(deviceName, "Max")) {
    return DeviceTier::MacUltra;
  }
  if (std::strstr(deviceName, "iPhone") || std::strstr(deviceName, "iPad") ||
      std::strstr(deviceName, "Apple A")) {
    return DeviceTier::Phone;
  }
  return DeviceTier::MacBase;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
