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
using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpArgminTest : public OperatorTest {
 protected:
  Tensor& op_argmin_out(
      const Tensor& in,
      optional<int64_t> dim,
      bool keepdim,
      Tensor& out) {
    return torch::executor::aten::argmin_outf(context_, in, dim, keepdim, out);
  }

  template <ScalarType dtype>
  void test_argmin_dtype() {
    TensorFactory<ScalarType::Long> tfl;
    TensorFactory<dtype> tf_dtype;

    // clang-format off
    Tensor in = tf_dtype.make(
        { 2, 3, 4 },
        { 1, 4, 1, 6,
          5, 8, 5, 6,
          5, 3, 9, 2,

          3, 9, 1, 4,
          9, 7, 5, 5,
          7, 7, 6, 3 });

    Tensor out = tfl.zeros({2, 4});
    Tensor expected = tfl.make({2, 4}, {
        0, 2, 0, 2,
        0, 1, 0, 2 });
    Tensor ret = op_argmin_out(in, 1, false, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
    // clang-format on
  }
};

TEST_F(OpArgminTest, SanityCheck) {
#define TEST_ENTRY(ctype, dtype) test_argmin_dtype<ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpArgminTest, SanityCheckNullDim) {
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
  Tensor expected = tf.make({}, {2});

  optional<int64_t> dim;
  Tensor ret = op_argmin_out(in, dim, false, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
  // clang-format on
}

TEST_F(OpArgminTest, FirstNaNWins) {
  TensorFactory<ScalarType::Float> tf_float;
  Tensor in = tf_float.make({4}, {1, NAN, -4, NAN});

  TensorFactory<ScalarType::Long> tf_long;
  Tensor out = tf_long.zeros({});
  Tensor expected = tf_long.make({}, {1});

  Tensor ret = op_argmin_out(in, {}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}
