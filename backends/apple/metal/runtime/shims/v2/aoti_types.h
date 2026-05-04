/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SlimTensor-flavored types for the v2 AOTI Metal backend.
//
// The v2 backend uses SlimTensor (executorch::backends::aoti::slim::SlimTensor)
// as its tensor type. This header is the v2 analogue of types.h.

#pragma once

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace metal {

// common_shims_slim.h defines `using Tensor = slim::SlimTensor;` in the
// `executorch::backends::aoti` namespace. Re-export here so callers in
// `executorch::backends::metal` can write `Tensor` unqualified.
using executorch::runtime::Error;
using executorch::backends::aoti::Tensor;
using executorch::backends::aoti::AOTIRuntimeError;
using executorch::backends::aoti::AOTITorchError;

extern "C" {

// AOTI passes opaque tensor handles across the C ABI. In v2, these are
// SlimTensor pointers.
using AOTITensorHandle = Tensor*;

} // extern "C"

// Map int32_t dtype code (PyTorch convention) to slim::c10::ScalarType.
// Mirrors aoti::dtype_to_scalar_type but returns slim's enum instead of
// executorch::aten::ScalarType.
inline executorch::backends::aoti::slim::c10::ScalarType
dtype_to_c10_scalar_type(int32_t dtype) {
  using SST = executorch::backends::aoti::slim::c10::ScalarType;
  // Enum values match PyTorch's standard dtype encoding, so a value-cast is
  // safe for the supported set. Unsupported values (e.g. Half=5) will fail
  // SlimTensor's check_supportive assertion downstream.
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

} // namespace metal
} // namespace backends
} // namespace executorch
