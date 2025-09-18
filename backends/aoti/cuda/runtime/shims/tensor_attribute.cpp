/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/cuda/runtime/shims/tensor_attribute.h>

namespace executorch {
namespace backends {
namespace aoti {

extern "C" {

// Device type functions for tensor attributes
AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type) {
  // All tensors in aoti-cuda delegate are on CUDA
  *ret_device_type = aoti_torch_device_type_cuda();
  return Error::Ok;
}

// Device type constants
__attribute__((__visibility__("default"))) int32_t
aoti_torch_device_type_cuda() {
  // Let's say cuda is 1 for ET as well
  return 1;
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
