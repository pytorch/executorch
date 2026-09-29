// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/NativeModule.h>

#include <memory>

#include <executorch/backends/native/extension/module/test/TestData.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <gtest/gtest.h>

namespace executorch::extension::native_module {
namespace {

class NativeModuleLoadTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch_native_module_ptn_link_anchor();
  }
};

TEST_F(NativeModuleLoadTest, Load_ValidPackage_ExposesMetadataWithoutEngine) {
  const std::vector<uint8_t> bytes = testing::make_tensor_package();
  Module module(std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));

  EXPECT_EQ(module.load(), runtime::Error::Ok);
  ASSERT_TRUE(module.format().ok());
  EXPECT_EQ(*module.format(), Module::Format::Ptn);
  EXPECT_EQ(module.program(), nullptr);
  ASSERT_TRUE(module.num_methods().ok());
  EXPECT_EQ(*module.num_methods(), 1);
  ASSERT_TRUE(module.method_names().ok());
  EXPECT_EQ(
      *module.method_names(), (std::unordered_set<std::string>{"forward"}));

  const auto meta = module.method_meta("forward");
  ASSERT_TRUE(meta.ok());
  EXPECT_STREQ(meta->name(), "forward");
  ASSERT_TRUE(meta->input_tensor_meta(0).ok());
  EXPECT_TRUE(meta->input_tensor_meta(0)->name().empty());
  ASSERT_TRUE(meta->output_tensor_meta(0).ok());
  EXPECT_TRUE(meta->output_tensor_meta(0)->name().empty());

  EXPECT_EQ(module.load_method("forward"), runtime::Error::NotSupported);
  EXPECT_FALSE(module.is_method_loaded("forward"));
}

TEST_F(NativeModuleLoadTest, Load_InvalidPackage_DoesNotPublishState) {
  const std::vector<uint8_t> bytes{'P', 'K'};
  Module module(std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));

  EXPECT_EQ(module.load(), runtime::Error::InvalidProgram);
  EXPECT_FALSE(module.is_loaded());
  EXPECT_EQ(module.load(), runtime::Error::InvalidProgram);
  EXPECT_FALSE(module.is_loaded());
}

TEST_F(NativeModuleLoadTest, Load_InternalConsistencyVerifiesConstants) {
  const std::vector<uint8_t> bytes =
      testing::make_tensor_package_with_bad_constant_checksum();
  Module minimal(
      std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));
  Module verified(
      std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));

  EXPECT_EQ(
      minimal.load(runtime::Program::Verification::Minimal),
      runtime::Error::Ok);
  EXPECT_EQ(
      verified.load(runtime::Program::Verification::InternalConsistency),
      runtime::Error::InvalidProgram);
}

} // namespace
} // namespace executorch::extension::native_module
