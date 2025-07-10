/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/bundled_module.h>
#include <gtest/gtest.h>

using namespace ::executorch::extension::ET_BUNDLED_MODULE_NAMESPACE;
using namespace ::executorch::runtime;

class BundledModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string resources_path;
    if (const char* env = std::getenv("RESOURCES_PATH")) {
      resources_path = env;
    }
    pte_path_ = std::getenv("ET_MODULE_PTE_PATH");
    bpte_path_ = resources_path + "/bundled_program.bpte";
  }

  static inline std::string bpte_path_;
  static inline std::string pte_path_;
};

TEST_F(BundledModuleTest, TestExecute) {
  auto bundled_module_output = BundledModule::from_file(bpte_path_.c_str());
  EXPECT_EQ(bundled_module_output.error(), Error::Ok);
  auto& bundled_module = bundled_module_output.get();

  auto outputs = bundled_module->execute("forward", /*testset_idx=*/0);
  EXPECT_EQ(bundled_module->Module::is_loaded(), true);
  EXPECT_EQ(outputs.error(), Error::Ok);

  auto status =
      bundled_module->verify_method_outputs("forward", /*testset_idx=*/0);
  EXPECT_EQ(status, Error::Ok);
}

TEST_F(BundledModuleTest, TestNonExistBPFile) {
  auto bundled_module_output =
      BundledModule::from_file("/path/to/nonexistent/file.bpte");
  EXPECT_EQ(bundled_module_output.error(), Error::AccessFailed);
}

TEST_F(BundledModuleTest, TestNonBPFile) {
  auto bundled_module_output = BundledModule::from_file(pte_path_.c_str());
  EXPECT_EQ(bundled_module_output.error(), Error::Ok);

  auto& bundled_module = bundled_module_output.get();

  auto outputs = bundled_module->execute("forward", /*testset_idx=*/0);
  EXPECT_EQ(bundled_module->Module::is_loaded(), false);
  EXPECT_EQ(outputs.error(), Error::InvalidArgument);

  auto status =
      bundled_module->verify_method_outputs("forward", /*testset_idx=*/0);
  EXPECT_EQ(status, Error::InvalidArgument);
}

TEST_F(BundledModuleTest, TestExecuteInvalidMethod) {
  auto bundled_module_output = BundledModule::from_file(bpte_path_.c_str());
  EXPECT_EQ(bundled_module_output.error(), Error::Ok);
  auto& bundled_module = bundled_module_output.get();

  auto outputs =
      bundled_module->execute("non_existent_method", /*testset_idx=*/0);
  EXPECT_EQ(outputs.error(), Error::InvalidArgument);
}

TEST_F(BundledModuleTest, TestExecuteInvalidIdx) {
  auto bundled_module_output = BundledModule::from_file(bpte_path_.c_str());
  EXPECT_EQ(bundled_module_output.error(), Error::Ok);
  auto& bundled_module = bundled_module_output.get();

  auto outputs = bundled_module->execute("forward", /*testset_idx=*/10000);
  EXPECT_EQ(outputs.error(), Error::InvalidArgument);
}

TEST_F(BundledModuleTest, TestVerifyInvalidMethod) {
  auto bundled_module_output = BundledModule::from_file(bpte_path_.c_str());
  EXPECT_EQ(bundled_module_output.error(), Error::Ok);
  auto& bundled_module = bundled_module_output.get();

  auto outputs = bundled_module->execute("forward", /*testset_idx=*/0);
  EXPECT_EQ(bundled_module->Module::is_loaded(), true);
  EXPECT_EQ(outputs.error(), Error::Ok);

  auto status = bundled_module->verify_method_outputs(
      "non_existent_method", /*testset_idx=*/0);
  EXPECT_EQ(status, Error::InvalidArgument);
}

TEST_F(BundledModuleTest, TestVerifyInvalidIdx) {
  auto bundled_module_output = BundledModule::from_file(bpte_path_.c_str());
  EXPECT_EQ(bundled_module_output.error(), Error::Ok);
  auto& bundled_module = bundled_module_output.get();

  auto outputs = bundled_module->execute("forward", /*testset_idx=*/0);
  EXPECT_EQ(bundled_module->Module::is_loaded(), true);
  EXPECT_EQ(outputs.error(), Error::Ok);

  auto status =
      bundled_module->verify_method_outputs("forward", /*testset_idx=*/10000);
  EXPECT_EQ(status, Error::InvalidArgument);
}
