/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace aoti {

using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

extern "C" {

// Type definitions
using AOTITensorHandle = Tensor*;
using AOTITorchError = Error;

// Utility function for printing tensor information
void aoti_torch_print_tensor_handle(AtenTensorHandle self, const char* msg);

// Cleanup function for tensor output file (called during backend destruction)
void cleanup_aoti_tensor_output();

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
