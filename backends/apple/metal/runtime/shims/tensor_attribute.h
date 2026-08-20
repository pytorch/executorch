/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/apple/metal/runtime/shims/types.h>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// Metal-specific device type function
int32_t aoti_torch_device_type_mps();

// Override aoti_torch_get_device_type to return MPS device type
AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type);

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
