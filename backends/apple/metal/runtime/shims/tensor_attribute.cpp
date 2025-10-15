/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/shims/tensor_attribute.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <iostream>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// Metal-specific device type constant
__attribute__((__visibility__("default"))) int32_t
aoti_torch_device_type_mps() {
  return 13; // Consistent with c10/core/DeviceType.h
}

// Override aoti_torch_get_device_type to return MPS device type
AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type) {
  *ret_device_type = aoti_torch_device_type_mps();
  return Error::Ok;
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
