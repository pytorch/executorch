/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <array>
#include <memory>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

namespace executorch::extension {
namespace {

TEST(PtnNoHooksTest, Load_PtnSource_ReturnsNotSupported) {
  runtime::runtime_init();
  const std::array<uint8_t, 2> bytes{'P', 'K'};
  Module module(std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));

  EXPECT_EQ(module.load(), runtime::Error::NotSupported);
  EXPECT_FALSE(module.is_loaded());
}

} // namespace
} // namespace executorch::extension
