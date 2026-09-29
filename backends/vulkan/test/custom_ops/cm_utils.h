/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace executorch {
namespace vulkan {
namespace prototyping {

// Query and print VK_KHR_cooperative_matrix properties from the device.
// Shows supported M/N/K tile sizes, component types, and scopes.
void queryCooperativeMatrixProperties();

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
