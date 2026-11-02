// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace executorch {
namespace vulkan {
namespace prototyping {

// Prints the active Vulkan adapter's properties and the cooperative matrix
// configurations it exposes.
void print_device_info();

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
