// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/NativeModule.h>

#include <memory>
#include <utility>
#include <vector>

#include <executorch/backends/native/extension/module/test/TestData.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/backend/options.h>
#include <gtest/gtest.h>

namespace executorch::extension::native_module {
namespace {

class ShortReadLoader final : public runtime::DataLoader {
 public:
  explicit ShortReadLoader(std::vector<uint8_t> bytes)
      : bytes_(std::move(bytes)) {}

  runtime::Result<runtime::FreeableBuffer>
  load(size_t offset, size_t size, const SegmentInfo&) const override {
    if (offset > bytes_.size() || size > bytes_.size() - offset) {
      return runtime::Error::InvalidArgument;
    }
    const size_t returned_size = size > 2 ? size - 1 : size;
    return runtime::FreeableBuffer(
        bytes_.data() + offset, returned_size, /*free_fn=*/nullptr);
  }

  runtime::Result<size_t> size() const override {
    return bytes_.size();
  }

 private:
  std::vector<uint8_t> bytes_;
};

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

TEST_F(NativeModuleLoadTest, Load_NonEmptyBackendOptions_DoesNotPublishState) {
  const std::vector<uint8_t> bytes = testing::make_tensor_package();
  Module module(std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));
  runtime::LoadBackendOptionsMap backend_options;
  runtime::BackendOptions<1> options;
  ASSERT_EQ(
      options.set_option("unsupported", /*value=*/true), runtime::Error::Ok);
  ASSERT_EQ(
      backend_options.set_options("engine", options.view()),
      runtime::Error::Ok);

  EXPECT_EQ(module.load(backend_options), runtime::Error::NotSupported);
  EXPECT_FALSE(module.is_loaded());
  EXPECT_EQ(module.load(), runtime::Error::Ok);
  EXPECT_TRUE(module.is_loaded());
}

TEST_F(NativeModuleLoadTest, Load_ExternalDataLoader_IsRejected) {
  const std::vector<uint8_t> bytes = testing::make_tensor_package();
  const std::vector<uint8_t> external_data{0};
  Module module(
      std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()),
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*event_tracer=*/nullptr,
      std::make_unique<BufferDataLoader>(
          external_data.data(), external_data.size()));

  EXPECT_EQ(module.load(), runtime::Error::InvalidArgument);
  EXPECT_FALSE(module.is_loaded());
}

TEST_F(NativeModuleLoadTest, Load_UnsupportedVersion_DoesNotPublishState) {
  const std::vector<uint8_t> bytes =
      testing::make_tensor_package_with_version("2.0");
  Module module(std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));

  EXPECT_EQ(module.load(), runtime::Error::NotSupported);
  EXPECT_FALSE(module.is_loaded());
}

TEST_F(NativeModuleLoadTest, Load_ShortDataLoaderRead_DoesNotPublishState) {
  Module module(
      std::make_unique<ShortReadLoader>(testing::make_tensor_package()));

  EXPECT_EQ(module.load(), runtime::Error::InvalidProgram);
  EXPECT_FALSE(module.is_loaded());
}

} // namespace
} // namespace executorch::extension::native_module
