/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

namespace torch::executor::testing {
class BinaryLogicalOpTest : public OperatorTest {
 protected:
  // Implement this to call the torch::executor::aten::op_outf function for the
  // op.
  virtual exec_aten::Tensor& op_out(
      const exec_aten::Tensor& lhs,
      const exec_aten::Tensor& rhs,
      exec_aten::Tensor& out) = 0;

  // Scalar reference implementation of the function in question for testing.
  virtual double op_reference(double x, double y) const = 0;

  template <
      exec_aten::ScalarType IN_DTYPE,
      exec_aten::ScalarType IN_DTYPE2,
      exec_aten::ScalarType OUT_DTYPE>
  void test_op_out() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<IN_DTYPE2> tf_in2;
    TensorFactory<OUT_DTYPE> tf_out;

    exec_aten::Tensor out = tf_out.zeros({1, 4});

    using CTYPE1 = typename decltype(tf_in)::ctype;
    std::vector<CTYPE1> test_vector1 = {0, CTYPE1(-1), CTYPE1(0), CTYPE1(31)};

    using CTYPE2 = typename decltype(tf_in2)::ctype;
    std::vector<CTYPE2> test_vector2 = {
        CTYPE2(0),
        CTYPE2(0),
        CTYPE2(15),
        CTYPE2(12),
    };

    std::vector<typename decltype(tf_out)::ctype> expected_vector;
    for (int ii = 0; ii < test_vector1.size(); ++ii) {
      expected_vector.push_back(
          op_reference(test_vector1[ii], test_vector2[ii]));
    }

    op_out(
        tf_in.make({1, 4}, test_vector1),
        tf_in2.make({1, 4}, test_vector2),
        out);

    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 4}, expected_vector));
  }

  void test_all_dtypes();
};

#define IMPLEMENT_BINARY_LOGICAL_OP_TEST(TestName) \
  TEST_F(TestName, SimpleTestAllTypes) {           \
    test_all_dtypes();                             \
  }
} // namespace torch::executor::testing
