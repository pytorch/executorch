/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Backend-shared types used by MetalStream, MetalKernel, MetalKernelCompiler,
// and op implementations.
//
// HISTORICAL NOTE: this file used to define abstract base classes
// `GpuStream`, `GpuKernel`, `GpuKernelCompiler` that pretended to be
// backend-agnostic. They weren't — terminology and assumptions baked in
// (ICB, metallib, MTLComputePipelineState references, etc.) made them
// Metal-specific. With Metal as the only impl, the abstraction added
// boilerplate without value, so the classes were collapsed into their
// `Metal*` concrete versions. This header keeps the genuinely-shared
// types — argument representation, kernel grid/block layout, dtype
// helpers — that are used across MetalStream / op files.

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
    case ScalarType::Float: return "f32";
    case ScalarType::Half: return "f16";
    case ScalarType::BFloat16: return "bf16";
    case ScalarType::Int: return "i32";
    case ScalarType::Long: return "i64";
    case ScalarType::Bool: return "bool";
    default:
      ET_CHECK_MSG(false, "Unsupported dtype for GPU kernel: %d", static_cast<int>(dtype));
      return nullptr;
  }
}

/// Check if dtype is a floating point type
inline bool isFloatingPoint(ScalarType dtype) {
  return dtype == ScalarType::Float ||
         dtype == ScalarType::Half ||
         dtype == ScalarType::BFloat16;
}

//===----------------------------------------------------------------------===//
// Arg - Unified argument type for dispatch
//===----------------------------------------------------------------------===//

struct Arg {
  enum Type { BUFFER, SCALAR_INT, SCALAR_FLOAT, TENSOR } type;

  union {
    struct { void* ptr; size_t size; } buffer;
    int64_t scalar_int;
    double scalar_float;
    struct {
      void* ptr;           // Data pointer
      size_t size;         // Total size in bytes
      int64_t dims[8];     // Dimension sizes (up to 8D)
      int64_t strides[8];  // Strides in elements
      int32_t rank;        // Number of dimensions
      int32_t dtype;       // Data type (MTLTensorDataType under Metal)
    } tensor;
  };

  Arg(void* ptr, size_t size) : type(BUFFER) {
    buffer.ptr = ptr;
    buffer.size = size;
  }

  Arg(int64_t val) : type(SCALAR_INT), scalar_int(val) {}
  Arg(int32_t val) : type(SCALAR_INT), scalar_int(val) {}
  Arg(uint32_t val) : type(SCALAR_INT), scalar_int(val) {}

  Arg(float val) : type(SCALAR_FLOAT), scalar_float(val) {}
  Arg(double val) : type(SCALAR_FLOAT), scalar_float(val) {}

  static Arg Tensor2D(void* ptr, size_t size, int64_t dim0, int64_t dim1, int32_t dtype) {
    Arg arg;
    arg.type = TENSOR;
    arg.tensor.ptr = ptr;
    arg.tensor.size = size;
    arg.tensor.rank = 2;
    arg.tensor.dtype = dtype;
    arg.tensor.dims[0] = dim0;
    arg.tensor.dims[1] = dim1;
    arg.tensor.strides[0] = dim1;  // Row-major
    arg.tensor.strides[1] = 1;
    return arg;
  }

private:
  Arg() : type(BUFFER) {}
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
