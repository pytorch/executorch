/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Shared Metal-shader-side helpers for index decoding (strided / broadcast
// access). Mirrors mlx/backend/metal/kernels/utils.h.
//
// Usage from a host .mm:
//   #include "metal_v2/kernels/Accessors.h"
//   const char* MyOp::kernelSource() const {
//     static const std::string source =
//         std::string(kAccessorsMetalSource) + R"(
//       // ... your kernel ...
//     )";
//     return source.c_str();
//   }
//
// All helpers use `int` strides (matches our host conversion in OpUtils
// where strides are i32 by the time they reach the kernel).

namespace executorch {
namespace backends {
namespace metal_v2 {

inline constexpr const char* kAccessorsMetalSource = R"METAL(
//===----------------------------------------------------------------------===//
// WorkPerThread<T>
//
// How many elements each thread should process in elementwise kernels.
// Smaller dtypes -> more elements/thread for better memory throughput.
// Mirrors mlx::WorkPerThread.
//
// MUST stay in sync with the host-side OpUtils::workPerThread(dtype).
//===----------------------------------------------------------------------===//

template <typename T> struct WorkPerThread { static constant constexpr int n = 4; };
template <>           struct WorkPerThread<half>   { static constant constexpr int n = 8; };
template <>           struct WorkPerThread<char>   { static constant constexpr int n = 8; };
template <>           struct WorkPerThread<uchar>  { static constant constexpr int n = 8; };
template <>           struct WorkPerThread<short>  { static constant constexpr int n = 8; };

//===----------------------------------------------------------------------===//
// elemToLoc family
//
// Convert a flat element index (or a multi-dim grid position) into a strided
// byte/element offset into a tensor's underlying buffer. Mirrors MLX's
// elem_to_loc_* in mlx/backend/metal/kernels/utils.h.
//
// Convention: shape[ndim - 1] is the innermost dimension.
// Strides may be 0 to express broadcasting along that dimension.
//===----------------------------------------------------------------------===//

// Generic N-D: flat elem -> strided offset.
inline int elemToLoc(
    uint elem,
    constant const int* shape,
    constant const int* strides,
    int ndim) {
  int loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += int(elem % uint(shape[i])) * strides[i];
    elem /= uint(shape[i]);
  }
  return loc;
}

// 1-D specialization (no shape needed; just a stride).
inline int elemToLoc1(uint elem, int stride) {
  return int(elem) * stride;
}

// 2-D specialization. elem.x = innermost (cols), elem.y = outer (rows).
// strides[0] = outer stride, strides[1] = inner stride (matches our
// outermost-first convention from OpUtils::collapseContiguousDims).
inline int elemToLoc2(uint2 elem, constant const int strides[2]) {
  return int(elem.x) * strides[1] + int(elem.y) * strides[0];
}

// 3-D specialization. elem.x = innermost, elem.y = middle, elem.z = outer.
inline int elemToLoc3(uint3 elem, constant const int strides[3]) {
  return int(elem.x) * strides[2] + int(elem.y) * strides[1] +
         int(elem.z) * strides[0];
}

//===----------------------------------------------------------------------===//
// elemToLocBinary / elemToLocTernary
//
// Decode the same flat elem against multiple inputs at once (binary/ternary
// general kernels). Returns one offset per input.
//===----------------------------------------------------------------------===//

inline int2 elemToLocBinary(
    uint elem,
    constant const int* shape,
    constant const int* a_strides,
    constant const int* b_strides,
    int ndim) {
  int2 loc = int2(0, 0);
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int dim = int(elem % uint(shape[i]));
    loc.x += dim * a_strides[i];
    loc.y += dim * b_strides[i];
    elem /= uint(shape[i]);
  }
  return loc;
}

inline int3 elemToLocTernary(
    uint elem,
    constant const int* shape,
    constant const int* a_strides,
    constant const int* b_strides,
    constant const int* c_strides,
    int ndim) {
  int3 loc = int3(0, 0, 0);
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int dim = int(elem % uint(shape[i]));
    loc.x += dim * a_strides[i];
    loc.y += dim * b_strides[i];
    loc.z += dim * c_strides[i];
    elem /= uint(shape[i]);
  }
  return loc;
}

)METAL";

} // namespace metal_v2
} // namespace backends
} // namespace executorch
