/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Shared types for the v2 AOTI Metal backend.

#pragma once

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace metal {

using executorch::runtime::Error;
using executorch::backends::aoti::Tensor;
using executorch::backends::aoti::AOTIRuntimeError;
using executorch::backends::aoti::AOTITorchError;

extern "C" {

// Opaque tensor handle the AOTI .so passes across the C ABI.
using AOTITensorHandle = Tensor*;

}  // extern "C"

// PyTorch dtype code → slim::c10::ScalarType. Unsupported codes return
// Undefined (e.g. Half=5; SlimTensor's check_supportive will fault).
inline executorch::backends::aoti::slim::c10::ScalarType
dtype_to_c10_scalar_type(int32_t dtype) {
  using SST = executorch::backends::aoti::slim::c10::ScalarType;
  switch (dtype) {
    case 0:  return SST::Byte;
    case 1:  return SST::Char;
    case 2:  return SST::Short;
    case 3:  return SST::Int;
    case 4:  return SST::Long;
    case 6:  return SST::Float;
    case 11: return SST::Bool;
    case 15: return SST::BFloat16;
    default: return SST::Undefined;
  }
}

}  // namespace metal
}  // namespace backends
}  // namespace executorch
