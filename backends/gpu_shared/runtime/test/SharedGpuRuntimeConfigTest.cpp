/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/gpu_shared/runtime/SharedGpuRuntimeConfig.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/span.h>

#include <gtest/gtest.h>
// Cppcheck's lint environment may not expand the gtest macros.
#ifndef TEST
#define TEST(test_suite_name, test_name) void test_suite_name##_##test_name()
#endif

using executorch::backends::gpu_shared::kSharedContextModeOption;
using executorch::backends::gpu_shared::kSharedContextTokenOption;
using executorch::backends::gpu_shared::kSharedGroupIdOption;
using executorch::backends::gpu_shared::parse_shared_gpu_runtime_config;
using executorch::backends::gpu_shared::SharedContextMode;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::Span;

namespace {

template <size_t N>
BackendInitContext make_context(BackendOptions<N>& options) {
  auto view = options.view();
  Span<const BackendOption> specs(view.data(), view.size());
  return BackendInitContext(nullptr, nullptr, nullptr, nullptr, specs);
}

// cppcheck-suppress unusedFunction
TEST(SharedGpuRuntimeConfigTest, UsesPersistentSharedDefaults) {
  BackendInitContext context(nullptr);

  auto result = parse_shared_gpu_runtime_config(context);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->token, "default");
  EXPECT_EQ(result->group_id, 0);
  EXPECT_EQ(result->context_mode, SharedContextMode::kLookupOrCreate);
}

// cppcheck-suppress unusedFunction
TEST(SharedGpuRuntimeConfigTest, ParsesRuntimeOptions) {
  BackendOptions<3> options;
  ASSERT_EQ(options.set_option(kSharedContextTokenOption, "scene0"), Error::Ok);
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "lookup_only"), Error::Ok);
  ASSERT_EQ(options.set_option(kSharedGroupIdOption, 7), Error::Ok);
  auto context = make_context(options);

  auto result = parse_shared_gpu_runtime_config(context);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->token, "scene0");
  EXPECT_EQ(result->group_id, 7);
  EXPECT_TRUE(result->lookup_only());
}

// cppcheck-suppress unusedFunction
TEST(SharedGpuRuntimeConfigTest, ParsesDisabledMode) {
  BackendOptions<1> options;
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "disabled"), Error::Ok);
  auto context = make_context(options);

  auto result = parse_shared_gpu_runtime_config(context);

  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result->enabled());
}

// cppcheck-suppress unusedFunction
TEST(SharedGpuRuntimeConfigTest, RejectsUnknownMode) {
  BackendOptions<1> options;
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "automatic"), Error::Ok);
  auto context = make_context(options);

  auto result = parse_shared_gpu_runtime_config(context);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

// cppcheck-suppress unusedFunction
TEST(SharedGpuRuntimeConfigTest, RejectsWrongRuntimeOptionType) {
  BackendOptions<1> options;
  ASSERT_EQ(options.set_option(kSharedGroupIdOption, "seven"), Error::Ok);
  auto context = make_context(options);

  auto result = parse_shared_gpu_runtime_config(context);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

// cppcheck-suppress unusedFunction
TEST(SharedGpuRuntimeConfigTest, RejectsEmptyTokenWhenEnabled) {
  BackendOptions<1> options;
  ASSERT_EQ(options.set_option(kSharedContextTokenOption, ""), Error::Ok);
  auto context = make_context(options);

  auto result = parse_shared_gpu_runtime_config(context);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

} // namespace
