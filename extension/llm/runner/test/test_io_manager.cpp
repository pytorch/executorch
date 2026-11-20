/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <gtest/gtest.h>

using executorch::extension::Module;
using executorch::extension::llm::io_manager_registry;
using executorch::extension::llm::IOManager;
using executorch::extension::llm::make_io_manager;
using executorch::extension::llm::register_io_manager;
using executorch::runtime::Error;

TEST(IOManagerRegistryTest, NoLabelUsesTheDefaultManager) {
  Module module("");
  auto result = make_io_manager(module, "");
  ASSERT_TRUE(result.ok());
  EXPECT_NE(*result, nullptr);
}

TEST(IOManagerRegistryTest, UnknownLabelReturnsAnError) {
  Module module("");
  EXPECT_EQ(
      make_io_manager(module, "unregistered_test_contract").error(),
      Error::NotSupported);
}

TEST(IOManagerRegistryTest, RegisteredFactoryIsUsedAndCannotBeReplaced) {
  Module module("");
  int calls = 0;
  const std::string label = "registry_test_contract";
  EXPECT_TRUE(register_io_manager(label, [&](Module& bound) {
    EXPECT_EQ(&bound, &module);
    ++calls;
    return std::make_unique<IOManager>(bound);
  }));
  EXPECT_FALSE(register_io_manager(
      label, [](Module&) { return std::unique_ptr<IOManager>{}; }));
  auto result = make_io_manager(module, label);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(calls, 1);
  io_manager_registry().erase(label);
}

TEST(IOManagerRegistryTest, NullFactoryResultReturnsAnError) {
  Module module("");
  const std::string label = "null_test_contract";
  ASSERT_TRUE(register_io_manager(
      label, [](Module&) { return std::unique_ptr<IOManager>{}; }));
  EXPECT_EQ(make_io_manager(module, label).error(), Error::InvalidState);
  io_manager_registry().erase(label);
}
