/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/Functions.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <algorithm>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::SizesType;
using exec_aten::StridesType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

// Note: This file is used for testing op_div for *portable kernel specific*.
// If your test case is generic and should be tested on all kernels, add it to
// executorch/kernels/test/op_div_test.cpp instead.

class OpDivScalarOutKernelTest : public OperatorTest {
 protected:
  Tensor& op_div_out_mode(
      const Tensor& a,
      const Tensor& b,
      exec_aten::optional<exec_aten::string_view> mode,
      Tensor& out) {
    return torch::executor::aten::div_outf(context_, a, b, mode, out);
  }
};

class OpDivScalarModeOutKernelTest : public OperatorTest {
 protected:
  Tensor& op_div_scalar_mode_out(
      const Tensor& a,
      const Scalar& b,
      exec_aten::optional<exec_aten::string_view> mode,
      Tensor& out) {
    return torch::executor::aten::div_outf(context_, a, b, mode, out);
  }
};

TEST_F(OpDivScalarOutKernelTest, SanityCheckModeTrunc) {
  TensorFactory<ScalarType::Int> tf_a;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf_out.zeros(sizes);

  op_div_out_mode(
      tf_a.make(sizes, {1, 2, 4, -9}),
      tf_a.make(sizes, {2, 2, 2, 2}),
      exec_aten::optional<exec_aten::string_view>("trunc"),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {0.0, 1.0, 2.0, -4.0}));
}

TEST_F(OpDivScalarOutKernelTest, SanityCheckModeFloor) {
  TensorFactory<ScalarType::Int> tf_a;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf_out.zeros(sizes);

  op_div_out_mode(
      tf_a.make(sizes, {1, 2, 4, -9}),
      tf_a.make(sizes, {2, 2, 2, 2}),
      exec_aten::optional<exec_aten::string_view>("floor"),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {0.0, 1.0, 2.0, -5.0}));
}

TEST_F(OpDivScalarModeOutKernelTest, SanityCheckModeTrunc) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_div_scalar_mode_out(
      tf.make(sizes, {1, 2, 4, -9}),
      2,
      exec_aten::optional<exec_aten::string_view>("trunc"),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {0, 1, 2, -4}));
}

TEST_F(OpDivScalarModeOutKernelTest, SanityCheckModeFloor) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_div_scalar_mode_out(
      tf.make(sizes, {1, 2, 4, -9}),
      2,
      exec_aten::optional<exec_aten::string_view>("floor"),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {0, 1, 2, -5}));
}
