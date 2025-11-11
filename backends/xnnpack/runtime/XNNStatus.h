/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <xnnpack.h>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

constexpr const uint16_t kOffset[] = {0, 19, 44, 73, 98, 131, 163};

constexpr const char* kData =
    "xnn_status_success\0"
    "xnn_status_uninitialized\0"
    "xnn_status_invalid_parameter\0"
    "xnn_status_invalid_state\0"
    "xnn_status_unsupported_parameter\0"
    "xnn_status_unsupported_hardware\0"
    "xnn_status_out_of_memory\0";

inline const char* xnn_status_to_string(enum xnn_status type) {
  assert(type <= xnn_status_out_of_memory);
  return &kData[kOffset[type]];
}

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
