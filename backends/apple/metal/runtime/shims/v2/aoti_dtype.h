/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Dtype + layout helpers shared across the v2 AOTI shim layer.

#pragma once

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace executorch {
namespace backends {
namespace metal {

// Both enums use the standard PyTorch dtype encoding; value-cast is safe.
inline executorch::aten::ScalarType to_aten_scalar_type(
    executorch::backends::aoti::slim::c10::ScalarType slim_dt) {
  return static_cast<executorch::aten::ScalarType>(static_cast<int>(slim_dt));
}

inline size_t dtype_to_bytes(int32_t dtype) {
  return executorch::backends::aoti::dtype_to_element_size(dtype);
}

// Standard PyTorch-style contiguous strides (in element units).
// For a degenerate shape with a 0-sized dim, strides for the higher
// dim collapse to 0 — same convention as torch.empty(N, 0).contiguous().
inline std::vector<int64_t> compute_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());
  if (sizes.empty()) return strides;
  int64_t stride = 1;
  for (ssize_t i = static_cast<ssize_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  return strides;
}

// Maximum tensor rank supported by StackTensorView (used by the
// MetalOpRegistry fallback path in aoti_fallback_op.mm). AOTI shader
// dispatch (aoti_kernel.mm) is rank-agnostic and not subject to this.
constexpr size_t kMaxTensorDim = 8;

}  // namespace metal
}  // namespace backends
}  // namespace executorch
