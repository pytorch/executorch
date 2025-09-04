/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdint>
#include "types.h"

namespace executorch {
namespace backends {
namespace aoti {

extern "C" {

// // Utility function for printing tensor information
// void aoti_torch_print_tensor_handle(AOTITensorHandle self, const char* msg);

// Cleanup function for tensor output file (called during backend destruction)
void cleanup_aoti_tensor_output();

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
