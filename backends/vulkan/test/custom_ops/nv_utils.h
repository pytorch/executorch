// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

namespace executorch {
namespace vulkan {
namespace prototyping {

// Query and print cooperative matrix properties supported by the device.
void queryCooperativeMatrixProperties();

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
