/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/ScalarOverflowTestMacros.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpHardTanhTest : public OperatorTest {
 protected:
  Tensor& op_hardtanh_out(
      const Tensor& self,
      const Scalar& min_val,
      const Scalar& max_val,
      Tensor& out) {
    return torch::executor::aten::hardtanh_outf(
        context_, self, min_val, max_val, out);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    CTYPE lowest_test_element;
    CTYPE lower_bound;
    if constexpr (std::numeric_limits<CTYPE>::is_signed) {
      lowest_test_element = -3;
      lower_bound = -2;
    } else {
      lowest_test_element = 0;
      lower_bound = 0;
    }
    Tensor in = tf.make({2, 2}, {lowest_test_element, 0, 1, 100});
    Tensor out = tf.zeros({2, 2});

    Tensor ret = op_hardtanh_out(in, lower_bound, 2, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {lower_bound, 0, 1, 2}));
  }

  template <ScalarType DTYPE>
  void expect_bad_scalar_value_dies(const Scalar& bad_value) {
    TensorFactory<DTYPE> tf;
    Tensor in = tf.ones({2, 2});
    Tensor out = tf.zeros({2, 2});

    // Test overflow for min parameter (using valid max)
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_hardtanh_out(in, bad_value, 1.0, out));

    // Test overflow for max parameter (using valid min)
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_hardtanh_out(in, -1.0, bad_value, out));
  }
};

TEST_F(OpHardTanhTest, SanityCheck) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

GENERATE_SCALAR_OVERFLOW_TESTS(OpHardTanhTest)
