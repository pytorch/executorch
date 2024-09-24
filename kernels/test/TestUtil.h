/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Kernel Test utilities.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

#ifdef USE_ATEN_LIB
/**
 * Ensure the kernel will fail when `_statement` is executed.
 * @param _statement Statement to execute.
 */
#define ET_EXPECT_KERNEL_FAILURE(_context, _statement) \
  EXPECT_ANY_THROW(_statement)

#define ET_EXPECT_KERNEL_FAILURE_WITH_MSG(_context, _statement, _matcher) \
  EXPECT_ANY_THROW(_statement)

#define ET_TEST_OP_SUPPORTS_MEMORY_FORMATS(                                  \
    tf, op, input_contiguous, expected_contiguous, channels_last_support)    \
  Tensor input_channels_last = tf.channels_last_like(input_contiguous);      \
  Tensor expected_channel_last = tf.channels_last_like(expected_contiguous); \
                                                                             \
  Tensor output_contiguous = tf.zeros_like(expected_contiguous);             \
  Tensor output_channels_last = tf.channels_last_like(output_contiguous);    \
                                                                             \
  Tensor ret = op(input_channels_last, output_channels_last);                \
  if (channels_last_support) {                                               \
    EXPECT_TENSOR_EQ(output_channels_last, expected_channel_last);           \
  } else {                                                                   \
    EXPECT_TENSOR_NE(output_channels_last, expected_channel_last);           \
  }                                                                          \
  EXPECT_TENSOR_EQ(output_channels_last, ret);

#else

#define ET_EXPECT_KERNEL_FAILURE(_context, _statement)              \
  do {                                                              \
    _statement;                                                     \
    expect_failure();                                               \
    if ((_context).failure_state() == torch::executor::Error::Ok) { \
      ET_LOG(Error, "Expected kernel failure but found success.");  \
      ADD_FAILURE();                                                \
    }                                                               \
  } while (false)

#define ET_EXPECT_KERNEL_FAILURE_WITH_MSG(_context, _statement, _msg) \
  do {                                                                \
    _statement;                                                       \
    expect_failure();                                                 \
    if ((_context).failure_state() == torch::executor::Error::Ok) {   \
      ET_LOG(Error, "Expected kernel failure but found success.");    \
      ADD_FAILURE();                                                  \
    }                                                                 \
  } while (false)

#define ET_TEST_OP_SUPPORTS_MEMORY_FORMATS(                                  \
    tf, op, input_contiguous, expected_contiguous, channels_last_support)    \
  Tensor input_channels_last = tf.channels_last_like(input_contiguous);      \
  Tensor expected_channel_last = tf.channels_last_like(expected_contiguous); \
                                                                             \
  Tensor output_contiguous = tf.zeros_like(expected_contiguous);             \
  Tensor output_channels_last = tf.channels_last_like(output_contiguous);    \
                                                                             \
  Tensor ret = op(input_channels_last, output_channels_last);                \
  if (channels_last_support) {                                               \
    EXPECT_TENSOR_EQ(output_channels_last, expected_channel_last);           \
  } else {                                                                   \
    EXPECT_TENSOR_NE(output_channels_last, expected_channel_last);           \
  }                                                                          \
  EXPECT_TENSOR_EQ(output_channels_last, ret);                               \
  ET_EXPECT_KERNEL_FAILURE(                                                  \
      context_, op(input_channels_last, output_contiguous));                 \
  ET_EXPECT_KERNEL_FAILURE(                                                  \
      context_, op(input_contiguous, output_channels_last));

#endif // USE_ATEN_LIB

/*
 * Common test fixture for kernel / operator-level tests. Provides
 * a runtime context object and verifies failure state post-execution.
 */
class OperatorTest : public ::testing::Test {
 public:
  OperatorTest() : expect_failure_(false) {}

  void SetUp() override {
    torch::executor::runtime_init();
  }

  void TearDown() override {
    // Validate error state.
    if (!expect_failure_) {
      EXPECT_EQ(context_.failure_state(), torch::executor::Error::Ok);
    } else {
      EXPECT_NE(context_.failure_state(), torch::executor::Error::Ok);
    }
  }

  void expect_failure() {
    expect_failure_ = true;
  }

 protected:
  executorch::runtime::KernelRuntimeContext context_;
  bool expect_failure_;
};
