/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Backend-shared types used by MetalStream, MetalKernel, MetalKernelCompiler,
// and op implementations. Currently just `uvec3` (3-D launch dim) and the
// `dtypeSuffix` / `isFloatingPoint` helpers; previous abstract base classes
// (`GpuStream`, `GpuKernel`, etc.) and the role-tagged Arg union were
// collapsed into the typed-setter API on MetalStream.

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/assert.h>
#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Use exec_aten::ScalarType directly.
using ScalarType = exec_aten::ScalarType;

//===----------------------------------------------------------------------===//
// Basic Types
//===----------------------------------------------------------------------===//

struct uvec3 {
  uint32_t x, y, z;
  uvec3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) : x(x), y(y), z(z) {}
};

//===----------------------------------------------------------------------===//
// Dtype helpers
//===----------------------------------------------------------------------===//

/// Convert ScalarType to kernel name suffix (f32, f16, bf16, etc.)
inline const char* dtypeSuffix(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Float:
      return "f32";
    case ScalarType::Half:
      return "f16";
    case ScalarType::BFloat16:
      return "bf16";
    case ScalarType::Int:
      return "i32";
    case ScalarType::Long:
      return "i64";
    case ScalarType::Bool:
      return "bool";
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported dtype for GPU kernel: %d",
          static_cast<int>(dtype));
      return nullptr;
  }
}

/// Check if dtype is a floating point type
inline bool isFloatingPoint(ScalarType dtype) {
  return dtype == ScalarType::Float || dtype == ScalarType::Half ||
      dtype == ScalarType::BFloat16;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
