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

class OpViewAsComplexTest : public OperatorTest {
 protected:
  Tensor& view_as_complex_copy_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::view_as_complex_copy_outf(
        context_, self, out);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void run_real_smoke_test() {
    constexpr auto COMPLEX_DTYPE = executorch::runtime::toComplexType(DTYPE);
    using ComplexType =
        typename executorch::runtime::ScalarTypeToCppType<COMPLEX_DTYPE>::type;

    TensorFactory<DTYPE> tf;
    TensorFactory<COMPLEX_DTYPE> tf_out;

    Tensor in = tf.make({2, 2, 2}, {3, 4, -1.7, 7.4, 5, -12, 8.3, 0.1});
    Tensor out = tf_out.make(
        {2, 2},
        {ComplexType(0, 0),
         ComplexType(0, 0),
         ComplexType(0, 0),
         ComplexType(0, 0)});
    Tensor expected = tf_out.make(
        {2, 2},
        {ComplexType(3, 4),
         ComplexType(-1.7, 7.4),
         ComplexType(5, -12),
         ComplexType(8.3, 0.1)});
    Tensor ret = view_as_complex_copy_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_empty_input() {
    constexpr auto COMPLEX_DTYPE = executorch::runtime::toComplexType(DTYPE);

    TensorFactory<DTYPE> tf;
    TensorFactory<COMPLEX_DTYPE> tf_out;

    Tensor in = tf.make(/*sizes=*/{3, 0, 4, 2}, /*data=*/{});
    Tensor out = tf_out.make(/*sizes=*/{3, 0, 4}, /*data=*/{});
    Tensor expected = tf_out.make(/*sizes=*/{3, 0, 4}, /*data=*/{});
    Tensor ret = view_as_complex_copy_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void one_dim_input() {
    constexpr auto COMPLEX_DTYPE = executorch::runtime::toComplexType(DTYPE);
    using ComplexType =
        typename executorch::runtime::ScalarTypeToCppType<COMPLEX_DTYPE>::type;

    TensorFactory<DTYPE> tf;
    TensorFactory<COMPLEX_DTYPE> tf_out;

    Tensor in = tf.make(/*sizes=*/{2}, {1.5, 2.5});
    Tensor out = tf_out.make(/*sizes=*/{}, {ComplexType(0, 0)});
    Tensor expected = tf_out.make(/*sizes=*/{}, {ComplexType(1.5, 2.5)});
    Tensor ret = view_as_complex_copy_out(in, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpViewAsComplexTest, RealSmokeTest) {
#define RUN_SMOKE_TEST(ctype, dtype)               \
  run_real_smoke_test<ctype, ScalarType::dtype>(); \
  test_empty_input<ctype, ScalarType::dtype>();    \
  one_dim_input<ctype, ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(RUN_SMOKE_TEST);
#undef RUN_SMOKE_TEST
}
