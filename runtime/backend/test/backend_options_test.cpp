/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;

class BackendOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
  BackendOptions<5> options; // Capacity of 5 for testing limits
};

// Test basic string functionality
TEST_F(BackendOptionsTest, HandlesStringOptions) {
  // Set and retrieve valid string
  options.set_option("backend_type", "GPU");
  const char* result = nullptr;
  EXPECT_EQ(options.get_option("backend_type", result), Error::Ok);
  EXPECT_STREQ(result, "GPU");

  // Update existing key
  options.set_option("backend_type", "CPU");
  EXPECT_EQ(options.get_option("backend_type", result), Error::Ok);
  EXPECT_STREQ(result, "CPU");
}

// Test boolean options
TEST_F(BackendOptionsTest, HandlesBoolOptions) {
  options.set_option("debug", true);
  bool debug = false;
  EXPECT_EQ(options.get_option("debug", debug), Error::Ok);
  EXPECT_TRUE(debug);

  // Test false value
  options.set_option("verbose", false);
  EXPECT_EQ(options.get_option("verbose", debug), Error::Ok);
  EXPECT_FALSE(debug);
}

// Test integer options
TEST_F(BackendOptionsTest, HandlesIntOptions) {
  options.set_option("num_threads", 256);
  int num_threads = 0;
  EXPECT_EQ(options.get_option("num_threads", num_threads), Error::Ok);
  EXPECT_EQ(num_threads, 256);
}

// Test error conditions
TEST_F(BackendOptionsTest, HandlesErrors) {
  // Non-existent key
  bool dummy_bool;
  EXPECT_EQ(options.get_option("missing", dummy_bool), Error::NotFound);

  // Type mismatch
  options.set_option("threshold", 100);
  const char* dummy_str = nullptr;
  EXPECT_EQ(options.get_option("threshold", dummy_str), Error::InvalidArgument);

  // Null value handling, should expect failure
  ET_EXPECT_DEATH(
      options.set_option("nullable", static_cast<const char*>(nullptr)), "");
}

// Test type-specific keys
TEST_F(BackendOptionsTest, EnforcesKeyTypes) {
  // Same key name - later set operations overwrite earlier ones
  options.set_option("flag", true);
  options.set_option("flag", 123); // Overwrites the boolean entry

  bool bval;
  int ival;

  // Boolean get should fail - type was overwritten to INT
  EXPECT_EQ(options.get_option("flag", bval), Error::InvalidArgument);

  // Integer get should succeed with correct value
  EXPECT_EQ(options.get_option("flag", ival), Error::Ok);
  EXPECT_EQ(ival, 123);
}

TEST_F(BackendOptionsTest, MutableOption) {
  int ival;
  options.set_option("flag", 0);
  // Integer get should succeed with correct value
  EXPECT_EQ(options.get_option("flag", ival), Error::Ok);
  EXPECT_EQ(ival, 0);

  options.view()[0].value = 123; // Overwrites the entry

  // Integer get should succeed with the updated value
  EXPECT_EQ(options.get_option("flag", ival), Error::Ok);
  EXPECT_EQ(ival, 123);
}

// Test copy constructor
TEST_F(BackendOptionsTest, CopyConstructor) {
  // Set up original option
  options.set_option("debug", true);

  // Create copy using copy constructor
  BackendOptions<5> copied_options(options);

  // Verify option was copied correctly
  bool debug_val;
  EXPECT_EQ(copied_options.get_option("debug", debug_val), Error::Ok);
  EXPECT_TRUE(debug_val);

  // Verify independence - modifying original doesn't affect copy
  options.set_option("debug", false);
  EXPECT_EQ(copied_options.get_option("debug", debug_val), Error::Ok);
  EXPECT_TRUE(debug_val); // Should still be true in copy

  // Verify independence - modifying copy doesn't affect original
  copied_options.set_option("debug", false);
  EXPECT_EQ(options.get_option("debug", debug_val), Error::Ok);
  EXPECT_FALSE(debug_val); // Should be false in original
}

// Test copy assignment operator
TEST_F(BackendOptionsTest, CopyAssignmentOperator) {
  // Set up original option
  options.set_option("enable_profiling", true);

  // Create another options object and assign to it
  BackendOptions<5> assigned_options;
  assigned_options.set_option("temp_option", false); // Add something first

  assigned_options = options;

  // Verify option was copied correctly
  bool profiling_val;
  EXPECT_EQ(
      assigned_options.get_option("enable_profiling", profiling_val),
      Error::Ok);
  EXPECT_TRUE(profiling_val);

  // Verify the temp_option was overwritten (not present in assigned object)
  bool temp_val;
  EXPECT_EQ(
      assigned_options.get_option("temp_option", temp_val), Error::NotFound);

  // Verify independence - modifying original doesn't affect assigned copy
  options.set_option("enable_profiling", false);
  EXPECT_EQ(
      assigned_options.get_option("enable_profiling", profiling_val),
      Error::Ok);
  EXPECT_TRUE(profiling_val); // Should still be true in assigned copy
}
