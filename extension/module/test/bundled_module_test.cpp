/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/extension/module/bundled_module.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace ::executorch::extension::ET_BUNDLED_MODULE_NAMESPACE;
using namespace ::executorch::runtime;
using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::runtime::testing::TensorFactory;

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

TEST(BundledProgramErrorStatsTest, HalfValuesAreFiniteAndCorrect) {
  TensorFactory<ScalarType::Half> tensor_factory;
  const auto expected = tensor_factory.make({2}, {Half(1.0f), Half(2.0f)});
  const auto actual = tensor_factory.make({2}, {Half(1.5f), Half(1.0f)});

  const auto stats =
      executorch::BUNDLED_PROGRAM_NAMESPACE::compute_tensor_error_stats(
          actual, expected);

  EXPECT_EQ(stats.status, Error::Ok);
  EXPECT_TRUE(std::isfinite(stats.mean_abs_error));
  EXPECT_TRUE(std::isfinite(stats.max_abs_error));
  EXPECT_TRUE(std::isfinite(stats.mean_relative_error));
  EXPECT_TRUE(std::isfinite(stats.max_relative_error));
  EXPECT_DOUBLE_EQ(stats.mean_abs_error, 0.75);
  EXPECT_DOUBLE_EQ(stats.max_abs_error, 1.0);
  EXPECT_NEAR(stats.mean_relative_error, 5.0 / 12.0, 1e-12);
  EXPECT_DOUBLE_EQ(stats.max_relative_error, 0.5);
}

TEST(BundledProgramErrorStatsTest, FloatValuesAreCorrect) {
  TensorFactory<ScalarType::Float> tensor_factory;
  const auto expected = tensor_factory.make({2}, {1.0f, 2.0f});
  const auto actual = tensor_factory.make({2}, {1.5f, 1.0f});

  const auto stats =
      executorch::BUNDLED_PROGRAM_NAMESPACE::compute_tensor_error_stats(
          actual, expected);

  EXPECT_EQ(stats.status, Error::Ok);
  EXPECT_DOUBLE_EQ(stats.mean_abs_error, 0.75);
  EXPECT_DOUBLE_EQ(stats.max_abs_error, 1.0);
  EXPECT_NEAR(stats.mean_relative_error, 5.0 / 12.0, 1e-12);
  EXPECT_DOUBLE_EQ(stats.max_relative_error, 0.5);
}

TEST(BundledProgramErrorStatsTest, MatchingNonFiniteValuesHaveZeroError) {
  TensorFactory<ScalarType::Float> tensor_factory;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto expected = tensor_factory.make({3}, {nan, inf, -inf});
  const auto actual = tensor_factory.make({3}, {nan, inf, -inf});

  const auto stats =
      executorch::BUNDLED_PROGRAM_NAMESPACE::compute_tensor_error_stats(
          actual, expected);

  EXPECT_EQ(stats.status, Error::Ok);
  EXPECT_DOUBLE_EQ(stats.mean_abs_error, 0.0);
  EXPECT_DOUBLE_EQ(stats.max_abs_error, 0.0);
  EXPECT_DOUBLE_EQ(stats.mean_relative_error, 0.0);
  EXPECT_DOUBLE_EQ(stats.max_relative_error, 0.0);
}

TEST(BundledProgramErrorStatsTest, FloatNonFiniteMismatchesHaveInfiniteError) {
  TensorFactory<ScalarType::Float> tensor_factory;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto expected = tensor_factory.make({3}, {0.0f, inf, -inf});
  const auto actual = tensor_factory.make({3}, {nan, -inf, 0.0f});

  const auto stats =
      executorch::BUNDLED_PROGRAM_NAMESPACE::compute_tensor_error_stats(
          actual, expected);

  EXPECT_EQ(stats.status, Error::Ok);
  EXPECT_TRUE(std::isinf(stats.mean_abs_error));
  EXPECT_TRUE(std::isinf(stats.max_abs_error));
  EXPECT_TRUE(std::isinf(stats.mean_relative_error));
  EXPECT_TRUE(std::isinf(stats.max_relative_error));
}

TEST(BundledProgramErrorStatsTest, HalfNonFiniteMismatchesHaveInfiniteError) {
  TensorFactory<ScalarType::Half> tensor_factory;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto expected = tensor_factory.make({2}, {Half(0.0f), Half(inf)});
  const auto actual = tensor_factory.make({2}, {Half(nan), Half(-inf)});

  const auto stats =
      executorch::BUNDLED_PROGRAM_NAMESPACE::compute_tensor_error_stats(
          actual, expected);

  EXPECT_EQ(stats.status, Error::Ok);
  EXPECT_TRUE(std::isinf(stats.mean_abs_error));
  EXPECT_TRUE(std::isinf(stats.max_abs_error));
  EXPECT_TRUE(std::isinf(stats.mean_relative_error));
  EXPECT_TRUE(std::isinf(stats.max_relative_error));
}

TEST(BundledProgramErrorStatsTest, UnsupportedDtypeIsRejected) {
  TensorFactory<ScalarType::Int> tensor_factory;
  const auto expected = tensor_factory.make({1}, {1});
  const auto actual = tensor_factory.make({1}, {2});

  const auto stats =
      executorch::BUNDLED_PROGRAM_NAMESPACE::compute_tensor_error_stats(
          actual, expected);

  EXPECT_EQ(stats.status, Error::NotSupported);
}

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
