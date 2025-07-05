/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpViewAsRealTest : public OperatorTest {
 protected:
  Tensor& view_as_real_copy_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::view_as_real_copy_outf(context_, self, out);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void run_complex_smoke_test() {
    TensorFactory<DTYPE> tf;
    constexpr auto REAL_DTYPE = executorch::runtime::toRealValueType(DTYPE);
    TensorFactory<REAL_DTYPE> tf_out;

    Tensor in = tf.make(
        {2, 2},
        {CTYPE(3, 4), CTYPE(-1.7, 7.4), CTYPE(5, -12), CTYPE(8.3, 0.1)});
    Tensor out = tf_out.zeros({2, 2, 2});
    Tensor expected =
        tf_out.make({2, 2, 2}, {3, 4, -1.7, 7.4, 5, -12, 8.3, 0.1});
    Tensor ret = view_as_real_copy_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }

  // Tests on tensors with 0 size
  template <typename CTYPE, ScalarType DTYPE>
  void test_empty_input() {
    TensorFactory<DTYPE> tf;
    constexpr auto REAL_DTYPE = executorch::runtime::toRealValueType(DTYPE);
    TensorFactory<REAL_DTYPE> tf_out;

    Tensor in = tf.make(/*sizes=*/{3, 0, 4}, /*data=*/{});
    Tensor out = tf_out.zeros({3, 0, 4, 2});
    Tensor expected = tf_out.make(/*sizes=*/{3, 0, 4, 2}, /*data=*/{});
    Tensor ret = view_as_real_copy_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }

  // Tests on 0-dim input tensors
  template <typename CTYPE, ScalarType DTYPE>
  void zero_dim_input() {
    TensorFactory<DTYPE> tf;
    constexpr auto REAL_DTYPE = executorch::runtime::toRealValueType(DTYPE);
    TensorFactory<REAL_DTYPE> tf_out;

    Tensor in = tf.make(/*sizes=*/{}, {CTYPE(0, 0)});
    Tensor out = tf_out.zeros({2});
    Tensor expected = tf_out.zeros(/*sizes=*/{2});
    Tensor ret = view_as_real_copy_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpViewAsRealTest, ComplexSmokeTest) {
#define RUN_SMOKE_TEST(ctype, dtype)                  \
  run_complex_smoke_test<ctype, ScalarType::dtype>(); \
  test_empty_input<ctype, ScalarType::dtype>();       \
  zero_dim_input<ctype, ScalarType::dtype>();
  ET_FORALL_COMPLEXH_TYPES(RUN_SMOKE_TEST);
#undef RUN_SMOKE_TEST
}
