/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdint>

namespace executorch::backends::cuda {

// Common using declarations for ExecutorTorch types
using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

extern "C" {

// Common AOTI type aliases
using AOTITorchError = Error;

// Device type functions for tensor attributes
AOTITorchError aoti_torch_get_device_type(
    Tensor* tensor,
    int32_t* ret_device_type);

// Device type constants
int32_t aoti_torch_device_type_cuda();

} // extern "C"

} // namespace executorch::backends::cuda
