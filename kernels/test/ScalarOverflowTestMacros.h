/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Macro to generate scalar overflow test cases for a given test suite.
// The test suite must have a method called expect_bad_scalar_value_dies
// that takes a template parameter for ScalarType and a Scalar value.
#define GENERATE_SCALAR_OVERFLOW_TESTS(TEST_SUITE_NAME)         \
  TEST_F(TEST_SUITE_NAME, ByteTensorTooLargeScalarDies) {       \
    /* Cannot be represented by a uint8_t. */                   \
    expect_bad_scalar_value_dies<ScalarType::Byte>(256);        \
  }                                                             \
                                                                \
  TEST_F(TEST_SUITE_NAME, CharTensorTooSmallScalarDies) {       \
    /* Cannot be represented by a int8_t. */                    \
    expect_bad_scalar_value_dies<ScalarType::Char>(-129);       \
  }                                                             \
                                                                \
  TEST_F(TEST_SUITE_NAME, ShortTensorTooLargeScalarDies) {      \
    /* Cannot be represented by a int16_t. */                   \
    expect_bad_scalar_value_dies<ScalarType::Short>(32768);     \
  }                                                             \
                                                                \
  TEST_F(TEST_SUITE_NAME, FloatTensorTooSmallScalarDies) {      \
    /* Cannot be represented by a float. */                     \
    expect_bad_scalar_value_dies<ScalarType::Float>(-3.41e+38); \
  }                                                             \
                                                                \
  TEST_F(TEST_SUITE_NAME, FloatTensorTooLargeScalarDies) {      \
    /* Cannot be represented by a float. */                     \
    expect_bad_scalar_value_dies<ScalarType::Float>(3.41e+38);  \
  }
