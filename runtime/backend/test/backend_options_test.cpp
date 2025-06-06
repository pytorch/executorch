/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/backend_options.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::BackendOptions;
using executorch::runtime::BoolKey;
using executorch::runtime::Error;
using executorch::runtime::IntKey;
using executorch::runtime::OptionKey;
using executorch::runtime::StrKey;

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
  options.set_option(StrKey("backend_type"), "GPU");
  const char* result = nullptr;
  EXPECT_EQ(options.get_option(StrKey("backend_type"), result), Error::Ok);
  EXPECT_STREQ(result, "GPU");

  // Update existing key
  options.set_option(StrKey("backend_type"), "CPU");
  EXPECT_EQ(options.get_option(StrKey("backend_type"), result), Error::Ok);
  EXPECT_STREQ(result, "CPU");
}

// Test boolean options
TEST_F(BackendOptionsTest, HandlesBoolOptions) {
  options.set_option(BoolKey("debug"), true);
  bool debug = false;
  EXPECT_EQ(options.get_option(BoolKey("debug"), debug), Error::Ok);
  EXPECT_TRUE(debug);

  // Test false value
  options.set_option(BoolKey("verbose"), false);
  EXPECT_EQ(options.get_option(BoolKey("verbose"), debug), Error::Ok);
  EXPECT_FALSE(debug);
}

// Test integer options
TEST_F(BackendOptionsTest, HandlesIntOptions) {
  options.set_option(IntKey("num_threads"), 256);
  int64_t num_threads = 0;
  EXPECT_EQ(options.get_option(IntKey("num_threads"), num_threads), Error::Ok);
  EXPECT_EQ(num_threads, 256);
}

// Test error conditions
TEST_F(BackendOptionsTest, HandlesErrors) {
  // Non-existent key
  bool dummy_bool;
  EXPECT_EQ(
      options.get_option(BoolKey("missing"), dummy_bool), Error::NotFound);

  // Type mismatch
  options.set_option(IntKey("threshold"), 100);
  const char* dummy_str = nullptr;
  EXPECT_EQ(
      options.get_option(StrKey("threshold"), dummy_str),
      Error::InvalidArgument);

  // Null value handling
  options.set_option(StrKey("nullable"), nullptr);
  EXPECT_EQ(options.get_option(StrKey("nullable"), dummy_str), Error::Ok);
  EXPECT_EQ(dummy_str, nullptr);
}

// Test capacity limits
TEST_F(BackendOptionsTest, HandlesCapacity) {
  // Use persistent storage for keys
  std::vector<std::string> keys = {"key0", "key1", "key2", "key3", "key4"};

  // Fill to capacity with persistent keys
  for (int i = 0; i < 5; i++) {
    options.set_option(IntKey(keys[i].c_str()), i);
  }

  // Verify all exist
  int64_t value;
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(options.get_option(IntKey(keys[i].c_str()), value), Error::Ok);
    EXPECT_EQ(value, i);
  }

  // Add beyond capacity - should fail
  const char* overflow_key = "overflow";
  options.set_option(IntKey(overflow_key), 99);
  EXPECT_EQ(options.get_option(IntKey(overflow_key), value), Error::NotFound);

  // Update existing within capacity
  options.set_option(IntKey(keys[2].c_str()), 222);
  EXPECT_EQ(options.get_option(IntKey(keys[2].c_str()), value), Error::Ok);
  EXPECT_EQ(value, 222);
}

// Test type-specific keys
TEST_F(BackendOptionsTest, EnforcesKeyTypes) {
  // Same key name - later set operations overwrite earlier ones
  options.set_option(BoolKey("flag"), true);
  options.set_option(IntKey("flag"), 123); // Overwrites the boolean entry

  bool bval;
  int64_t ival;

  // Boolean get should fail - type was overwritten to INT
  EXPECT_EQ(options.get_option(BoolKey("flag"), bval), Error::InvalidArgument);

  // Integer get should succeed with correct value
  EXPECT_EQ(options.get_option(IntKey("flag"), ival), Error::Ok);
  EXPECT_EQ(ival, 123);
}
