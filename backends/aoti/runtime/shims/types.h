/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace aoti {

// Common using declarations for ExecutorTorch types
using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

extern "C" {

// Common AOTI type aliases
// Note: AOTITensorHandle is aliased to Tensor* for ExecutorTorch compatibility
using AOTITensorHandle = Tensor*;
using AOTIRuntimeError = Error;
using AOTITorchError = Error;

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
