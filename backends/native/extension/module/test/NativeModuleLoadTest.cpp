// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/NativeModule.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
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

class TemporaryPackageFile final {
 public:
  explicit TemporaryPackageFile(const std::vector<uint8_t>& bytes)
      : path_(
            std::string(::testing::TempDir()) + "/native_module_load_test_" +
            std::to_string(next_id_.fetch_add(1)) + ".ptn") {
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot create temporary PTN package");
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) {
      throw std::runtime_error("cannot write temporary PTN package");
    }
  }

  ~TemporaryPackageFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::string& path() const {
    return path_;
  }

 private:
  static std::atomic<uint64_t> next_id_;
  std::string path_;
};

std::atomic<uint64_t> TemporaryPackageFile::next_id_{0};

void expect_same_tensor_signature(
    const runtime::TensorInfo& pte,
    const runtime::TensorInfo& ptn) {
  EXPECT_EQ(pte.scalar_type(), ptn.scalar_type());
  EXPECT_EQ(pte.nbytes(), ptn.nbytes());
  EXPECT_EQ(pte.name(), ptn.name());
  EXPECT_EQ(
      std::vector<int32_t>(pte.sizes().begin(), pte.sizes().end()),
      std::vector<int32_t>(ptn.sizes().begin(), ptn.sizes().end()));
  EXPECT_EQ(
      std::vector<uint8_t>(pte.dim_order().begin(), pte.dim_order().end()),
      std::vector<uint8_t>(ptn.dim_order().begin(), ptn.dim_order().end()));
}

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

TEST_F(NativeModuleLoadTest, MethodMeta_MatchesPteTensorSubset) {
  const char* pte_path = std::getenv("ET_MODULE_ADD_PATH");
  if (pte_path == nullptr) {
    GTEST_SKIP() << "ET_MODULE_ADD_PATH is unavailable";
  }
  Module pte(pte_path);
  const std::vector<uint8_t> ptn_bytes =
      testing::make_tensor_package({"forward"}, 2, {2, 2});
  Module ptn(
      std::make_unique<BufferDataLoader>(ptn_bytes.data(), ptn_bytes.size()));

  ASSERT_EQ(pte.load(), runtime::Error::Ok);
  ASSERT_EQ(ptn.load(), runtime::Error::Ok);
  const auto pte_meta = pte.method_meta("forward");
  const auto ptn_meta = ptn.method_meta("forward");
  ASSERT_TRUE(pte_meta.ok());
  ASSERT_TRUE(ptn_meta.ok());
  EXPECT_STREQ(pte_meta->name(), ptn_meta->name());
  ASSERT_EQ(pte_meta->num_inputs(), 3);
  ASSERT_EQ(ptn_meta->num_inputs(), 2);
  ASSERT_EQ(pte_meta->num_outputs(), ptn_meta->num_outputs());
  for (size_t i = 0; i < ptn_meta->num_inputs(); ++i) {
    const auto pte_tag = pte_meta->input_tag(i);
    const auto ptn_tag = ptn_meta->input_tag(i);
    ASSERT_TRUE(pte_tag.ok());
    ASSERT_TRUE(ptn_tag.ok());
    EXPECT_EQ(*pte_tag, *ptn_tag);
    const auto pte_tensor = pte_meta->input_tensor_meta(i);
    const auto ptn_tensor = ptn_meta->input_tensor_meta(i);
    ASSERT_TRUE(pte_tensor.ok());
    ASSERT_TRUE(ptn_tensor.ok());
    expect_same_tensor_signature(*pte_tensor, *ptn_tensor);
  }
  // ModuleAdd.pte has two tensor inputs followed by one scalar input.
  const auto extra_pte_input = pte_meta->input_tag(2);
  ASSERT_TRUE(extra_pte_input.ok());
  EXPECT_EQ(*extra_pte_input, runtime::Tag::Double);
  for (size_t i = 0; i < pte_meta->num_outputs(); ++i) {
    const auto pte_tag = pte_meta->output_tag(i);
    const auto ptn_tag = ptn_meta->output_tag(i);
    ASSERT_TRUE(pte_tag.ok());
    ASSERT_TRUE(ptn_tag.ok());
    EXPECT_EQ(*pte_tag, *ptn_tag);
    const auto pte_tensor = pte_meta->output_tensor_meta(i);
    const auto ptn_tensor = ptn_meta->output_tensor_meta(i);
    ASSERT_TRUE(pte_tensor.ok());
    ASSERT_TRUE(ptn_tensor.ok());
    expect_same_tensor_signature(*pte_tensor, *ptn_tensor);
  }
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

TEST_F(NativeModuleLoadTest, Load_UnsupportedMmapModes_ReturnNotSupported) {
  const TemporaryPackageFile file(testing::make_tensor_package());
  constexpr std::array<Module::LoadMode, 3> modes{
      Module::LoadMode::MmapUseMlock,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      Module::LoadMode::MmapUseMadvise,
  };

  for (const Module::LoadMode mode : modes) {
    Module module(file.path(), mode);
    EXPECT_EQ(module.load(), runtime::Error::NotSupported);
    EXPECT_FALSE(module.is_loaded());
  }
}

TEST_F(
    NativeModuleLoadTest,
    Load_UnsupportedModeTakesPrecedenceOverExternalData) {
  const TemporaryPackageFile file(testing::make_tensor_package());
  Module module(file.path(), file.path(), Module::LoadMode::MmapUseMlock);

  EXPECT_EQ(module.load(), runtime::Error::NotSupported);
  EXPECT_FALSE(module.is_loaded());
}

} // namespace
} // namespace executorch::extension::native_module
