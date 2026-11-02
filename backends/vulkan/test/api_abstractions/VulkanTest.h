// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

namespace vkcompute {

class VulkanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
    if (!api::available()) {
      GTEST_SKIP() << "Vulkan is unavailable on this test runner";
    }
  }
};

} // namespace vkcompute
