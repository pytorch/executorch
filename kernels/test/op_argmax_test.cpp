/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpArgmaxTest : public OperatorTest {
 protected:
  Tensor& op_argmax_out(
      const Tensor& in,
      optional<int64_t> dim,
      bool keepdim,
      Tensor& out) {
    return torch::executor::aten::argmax_outf(context_, in, dim, keepdim, out);
  }
};

TEST_F(OpArgmaxTest, SanityCheckLong) {
  TensorFactory<ScalarType::Long> tf;

  // clang-format off
  Tensor in = tf.make(
    { 2, 3, 4 },
    { 1, 4, 1, 6,
      5, 8, 5, 6,
      5, 3, 9, 2,

      3, 9, 1, 4,
      9, 7, 5, 5,
      7, 7, 6, 3 });

  Tensor out = tf.zeros({2, 4});
  Tensor expected = tf.make({2, 4}, {
    1, 1, 2, 0,
    1, 0, 2, 1 });
  Tensor ret = op_argmax_out(in, 1, false, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
  // clang-format on
}

TEST_F(OpArgmaxTest, SanityCheckShort) {
  TensorFactory<ScalarType::Long> tfl;
  TensorFactory<ScalarType::Short> tfs;

  // clang-format off
  Tensor in = tfs.make(
    { 2, 3, 4 },
    { 1, 4, 1, 6,
      5, 8, 5, 6,
      5, 3, 9, 2,

      3, 9, 1, 4,
      9, 7, 5, 5,
      7, 7, 6, 3 });

  Tensor out = tfl.zeros({2, 4});
  Tensor expected = tfl.make({2, 4}, {
    1, 1, 2, 0,
    1, 0, 2, 1 });
  Tensor ret = op_argmax_out(in, 1, false, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
  // clang-format on
}

TEST_F(OpArgmaxTest, SanityCheckNullDim) {
  TensorFactory<ScalarType::Long> tf;

  // clang-format off
  Tensor in = tf.make(
    { 2, 3, 4 },
    { 9, 4, 1, 6,
      5, 8, 5, 6,
      5, 3, 9, 2,

      3, 9, 1, 4,
      9, 7, 5, 5,
      7, 7, 6, 3 });

  Tensor out = tf.zeros({});
  Tensor expected = tf.make({}, {0});

  optional<int64_t> dim;
  Tensor ret = op_argmax_out(in, dim, false, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
  // clang-format on
}
