/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>

#include <gtest/gtest.h>

namespace executorch::backends::webgpu {
namespace {

TEST(WebGPUDeviceHeader, CompilesWithoutDirectDawnDependency) {
  WebGPUContext context;
  EXPECT_EQ(context.instance, nullptr);
  EXPECT_EQ(context.device, nullptr);
  EXPECT_EQ(context.queue, nullptr);
}

} // namespace
} // namespace executorch::backends::webgpu
