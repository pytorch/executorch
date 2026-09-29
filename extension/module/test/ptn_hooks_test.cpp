/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>
#include <executorch/extension/module/ptn_module.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

namespace executorch::extension {
namespace {

class FakePtnModule final : public native_module::internal::PtnModule {
 public:
  runtime::Result<size_t> num_methods() const override {
    return 1;
  }

  runtime::Result<std::unordered_set<std::string>> method_names()
      const override {
    return std::unordered_set<std::string>{"forward"};
  }

  runtime::Result<runtime::MethodMeta> method_meta(
      const std::string&) override {
    return runtime::Error::NotSupported;
  }

  runtime::Error load_method(const std::string& method_name) override {
    if (method_name != "forward") {
      return runtime::Error::InvalidArgument;
    }
    loaded_ = true;
    return runtime::Error::Ok;
  }

  bool unload_method(const std::string& method_name) override {
    if (method_name != "forward") {
      return false;
    }
    const bool was_loaded = loaded_;
    loaded_ = false;
    return was_loaded;
  }

  bool is_method_loaded(const std::string& method_name) const override {
    return method_name == "forward" && loaded_;
  }

  runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string&,
      const std::vector<runtime::EValue>&) override {
    return runtime::Error::NotSupported;
  }

  runtime::Error set_input(const std::string&, const runtime::EValue&, size_t)
      override {
    return runtime::Error::NotSupported;
  }

  runtime::Error set_inputs(
      const std::string&,
      const std::vector<runtime::EValue>&) override {
    return runtime::Error::NotSupported;
  }

  runtime::Error set_output(const std::string&, runtime::EValue, size_t)
      override {
    return runtime::Error::NotSupported;
  }

  runtime::Error set_outputs(
      const std::string&,
      const std::vector<runtime::EValue>&) override {
    return runtime::Error::NotSupported;
  }

  runtime::Result<std::vector<runtime::EValue>> get_outputs(
      const std::string&) override {
    return runtime::Error::NotSupported;
  }

  runtime::Result<runtime::EValue> get_output(const std::string&, size_t)
      override {
    return runtime::Error::NotSupported;
  }

 private:
  bool loaded_ = false;
};

runtime::Result<std::unique_ptr<native_module::internal::PtnModule>>
load_fake_ptn(runtime::DataLoader&, runtime::Program::Verification) {
  return std::unique_ptr<native_module::internal::PtnModule>(
      std::make_unique<FakePtnModule>());
}

const native_module::internal::PtnHooks kFakeHooks{load_fake_ptn};

class PtnHooksTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime::runtime_init();
    ASSERT_EQ(
        native_module::internal::register_ptn_hooks(kFakeHooks),
        runtime::Error::Ok);
  }
};

TEST_F(PtnHooksTest, Load_PtnSource_DispatchesMetadataAndMethodState) {
  const std::array<uint8_t, 2> bytes{'P', 'K'};
  Module module(std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));

  EXPECT_EQ(module.load(), runtime::Error::Ok);
  EXPECT_TRUE(module.is_loaded());
  ASSERT_TRUE(module.num_methods().ok());
  EXPECT_EQ(*module.num_methods(), 1);
  ASSERT_TRUE(module.method_names().ok());
  EXPECT_EQ(
      *module.method_names(), (std::unordered_set<std::string>{"forward"}));
  EXPECT_FALSE(module.is_method_loaded("forward"));
  EXPECT_EQ(module.load_method("forward"), runtime::Error::Ok);
  EXPECT_TRUE(module.is_method_loaded("forward"));
  EXPECT_TRUE(module.unload_method("forward"));
  EXPECT_FALSE(module.is_method_loaded("forward"));
}

TEST_F(PtnHooksTest, Register_SecondProvider_IsRejected) {
  EXPECT_EQ(
      native_module::internal::register_ptn_hooks(kFakeHooks),
      runtime::Error::AlreadyLoaded);
}

TEST_F(PtnHooksTest, PteOnlyAccessorsRejectWithoutCompiling) {
  const std::array<uint8_t, 2> bytes{'P', 'K'};
  Module module(std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));

  const auto method = module.method("forward");
  EXPECT_FALSE(method.ok());
  EXPECT_EQ(method.error(), runtime::Error::NotSupported);
  EXPECT_FALSE(module.is_method_loaded("forward"));
}

} // namespace
} // namespace executorch::extension
