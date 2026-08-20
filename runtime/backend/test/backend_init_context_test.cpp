/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::Span;

class BackendInitContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

// Test default constructor without runtime specs
TEST_F(BackendInitContextTest, DefaultConstructorNoRuntimeSpecs) {
  BackendInitContext context(nullptr);

  auto specs = context.runtime_specs();
  EXPECT_EQ(specs.size(), 0);
}

// Test constructor with runtime specs
TEST_F(BackendInitContextTest, ConstructorWithRuntimeSpecs) {
  BackendOptions<4> opts;
  opts.set_option("compute_unit", "cpu_and_gpu");
  opts.set_option("num_threads", 4);
  opts.set_option("enable_profiling", true);

  // Create a const span from the mutable view
  auto view = opts.view();
  Span<const BackendOption> const_span(view.data(), view.size());

  BackendInitContext context(
      nullptr, // runtime_allocator
      nullptr, // event_tracer
      "forward", // method_name
      nullptr, // named_data_map
      const_span // runtime_specs
  );

  auto specs = context.runtime_specs();
  EXPECT_EQ(specs.size(), 3);
}

// Test get_runtime_spec<bool> with valid key
TEST_F(BackendInitContextTest, GetRuntimeSpecBoolValid) {
  BackendOptions<2> opts;
  opts.set_option("enable_profiling", true);
  opts.set_option("debug_mode", false);

  auto view = opts.view();
  Span<const BackendOption> const_span(view.data(), view.size());

  BackendInitContext context(nullptr, nullptr, nullptr, nullptr, const_span);

  auto result1 = context.get_runtime_spec<bool>("enable_profiling");
  EXPECT_TRUE(result1.ok());
  EXPECT_TRUE(result1.get());

  auto result2 = context.get_runtime_spec<bool>("debug_mode");
  EXPECT_TRUE(result2.ok());
  EXPECT_FALSE(result2.get());
}

// Test get_runtime_spec<int> with valid key
TEST_F(BackendInitContextTest, GetRuntimeSpecIntValid) {
  BackendOptions<2> opts;
  opts.set_option("num_threads", 8);
  opts.set_option("batch_size", 32);

  auto view = opts.view();
  Span<const BackendOption> const_span(view.data(), view.size());

  BackendInitContext context(nullptr, nullptr, nullptr, nullptr, const_span);

  auto result1 = context.get_runtime_spec<int>("num_threads");
  EXPECT_TRUE(result1.ok());
  EXPECT_EQ(result1.get(), 8);

  auto result2 = context.get_runtime_spec<int>("batch_size");
  EXPECT_TRUE(result2.ok());
  EXPECT_EQ(result2.get(), 32);
}

// Test get_runtime_spec<const char*> with valid key
TEST_F(BackendInitContextTest, GetRuntimeSpecStringValid) {
  BackendOptions<2> opts;
  opts.set_option("compute_unit", "cpu_and_gpu");
  opts.set_option("cache_dir", "/tmp/cache");

  auto view = opts.view();
  Span<const BackendOption> const_span(view.data(), view.size());

  BackendInitContext context(nullptr, nullptr, nullptr, nullptr, const_span);

  auto result1 = context.get_runtime_spec<const char*>("compute_unit");
  EXPECT_TRUE(result1.ok());
  EXPECT_STREQ(result1.get(), "cpu_and_gpu");

  auto result2 = context.get_runtime_spec<const char*>("cache_dir");
  EXPECT_TRUE(result2.ok());
  EXPECT_STREQ(result2.get(), "/tmp/cache");
}

// Test get_runtime_spec<T> with non-existent key returns NotFound
TEST_F(BackendInitContextTest, GetRuntimeSpecNotFound) {
  BackendOptions<1> opts;
  opts.set_option("key", "value");

  auto view = opts.view();
  Span<const BackendOption> const_span(view.data(), view.size());

  BackendInitContext context(nullptr, nullptr, nullptr, nullptr, const_span);

  auto bool_result = context.get_runtime_spec<bool>("nonexistent");
  EXPECT_FALSE(bool_result.ok());
  EXPECT_EQ(bool_result.error(), Error::NotFound);

  auto int_result = context.get_runtime_spec<int>("nonexistent");
  EXPECT_FALSE(int_result.ok());
  EXPECT_EQ(int_result.error(), Error::NotFound);

  auto string_result = context.get_runtime_spec<const char*>("nonexistent");
  EXPECT_FALSE(string_result.ok());
  EXPECT_EQ(string_result.error(), Error::NotFound);
}

// Test get_runtime_spec<T> with wrong type returns InvalidArgument
TEST_F(BackendInitContextTest, GetRuntimeSpecTypeMismatch) {
  BackendOptions<3> opts;
  opts.set_option("bool_opt", true);
  opts.set_option("int_opt", 42);
  opts.set_option("string_opt", "hello");

  auto view = opts.view();
  Span<const BackendOption> const_span(view.data(), view.size());

  BackendInitContext context(nullptr, nullptr, nullptr, nullptr, const_span);

  // Try to get bool as int
  auto result1 = context.get_runtime_spec<int>("bool_opt");
  EXPECT_FALSE(result1.ok());
  EXPECT_EQ(result1.error(), Error::InvalidArgument);

  // Try to get int as string
  auto result2 = context.get_runtime_spec<const char*>("int_opt");
  EXPECT_FALSE(result2.ok());
  EXPECT_EQ(result2.error(), Error::InvalidArgument);

  // Try to get string as bool
  auto result3 = context.get_runtime_spec<bool>("string_opt");
  EXPECT_FALSE(result3.ok());
  EXPECT_EQ(result3.error(), Error::InvalidArgument);
}

// Test empty runtime specs
TEST_F(BackendInitContextTest, EmptyRuntimeSpecs) {
  Span<const BackendOption> empty_span;
  BackendInitContext context(nullptr, nullptr, nullptr, nullptr, empty_span);

  EXPECT_EQ(context.runtime_specs().size(), 0);

  // All lookups should return NotFound
  auto bool_result = context.get_runtime_spec<bool>("any_key");
  EXPECT_FALSE(bool_result.ok());
  EXPECT_EQ(bool_result.error(), Error::NotFound);
}

// Test that other context fields still work
TEST_F(BackendInitContextTest, OtherFieldsStillWork) {
  BackendOptions<1> opts;
  opts.set_option("key", "value");

  auto view = opts.view();
  Span<const BackendOption> const_span(view.data(), view.size());

  BackendInitContext context(
      nullptr, // runtime_allocator
      nullptr, // event_tracer
      "forward", // method_name
      nullptr, // named_data_map
      const_span // runtime_specs
  );

  EXPECT_EQ(context.get_runtime_allocator(), nullptr);
  EXPECT_EQ(context.event_tracer(), nullptr);
  EXPECT_STREQ(context.get_method_name(), "forward");
  EXPECT_EQ(context.get_named_data_map(), nullptr);
}
