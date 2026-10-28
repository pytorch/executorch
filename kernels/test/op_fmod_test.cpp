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
#include <limits>

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpFmodTest : public OperatorTest {
 protected:
  Tensor&
  op_fmod_tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::aten::fmod_outf(context_, self, other, out);
  }

  Tensor&
  op_fmod_scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    return torch::executor::aten::fmod_outf(context_, self, other, out);
  }

  template <ScalarType DTYPE>
  void test_integer_fmod() {
    TensorFactory<DTYPE> tf;
    Tensor self = tf.make({4}, {0, 7, 100, 127});
    Tensor out = tf.zeros({4});

    op_fmod_tensor_out(self, tf.make({4}, {3, 3, 7, 10}), out);
    EXPECT_TENSOR_EQ(out, tf.make({4}, {0, 1, 2, 7}));

    op_fmod_scalar_out(self, 3, out);
    EXPECT_TENSOR_EQ(out, tf.make({4}, {0, 1, 1, 1}));
  }
};

TEST_F(OpFmodTest, SmokeTest) {
  TensorFactory<ScalarType::Long> tfDouble;
  TensorFactory<ScalarType::Long> tfLong;
  TensorFactory<ScalarType::Int> tfInt;

  Tensor self = tfLong.full({2, 2}, 46);
  Tensor other = tfInt.full({2, 2}, 4);
  Tensor out = tfDouble.zeros({2, 2});
  Tensor out_expected = tfDouble.full({2, 2}, 2.0);
  op_fmod_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpFmodTest, Int64KeepsFullPrecision) {
  TensorFactory<ScalarType::Long> tf;
  // Values above 2^53, which a double cannot hold exactly.
  Tensor self = tf.make(
      {4},
      {(int64_t{1} << 62) + 3,
       -((int64_t{1} << 62) + 3),
       9007199254740993,
       std::numeric_limits<int64_t>::min()});
  Tensor out = tf.zeros({4});

  op_fmod_tensor_out(self, tf.make({4}, {10, 10, 2, -1}), out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {7, -7, 1, 0}));

  op_fmod_scalar_out(self, 10, out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {7, -7, 3, -8}));

  op_fmod_scalar_out(self, -1, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({4}));
}

TEST_F(OpFmodTest, AllIntegerDtypes) {
#define TEST_ENTRY(ctype, dtype) test_integer_fmod<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpFmodTest, NegativeIntegers) {
  TensorFactory<ScalarType::Char> tf;
  Tensor self = tf.make({4}, {-7, 7, -7, -128});
  Tensor out = tf.zeros({4});

  op_fmod_tensor_out(self, tf.make({4}, {3, -3, -3, -1}), out);
  EXPECT_TENSOR_EQ(out, tf.make({4}, {-1, 1, -1, 0}));
}

TEST_F(OpFmodTest, IntegerDivisionByZeroDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make({2}, {5, 6});
  Tensor out = tf.zeros({2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_fmod_tensor_out(self, tf.make({2}, {2, 0}), out));
  ET_EXPECT_KERNEL_FAILURE(context_, op_fmod_scalar_out(self, 0, out));
}

TEST_F(OpFmodTest, IntegerScalarIsCastToTensorDtype) {
  TensorFactory<ScalarType::Char> tf;
  Tensor self = tf.make({3}, {5, -7, 100});
  Tensor out = tf.zeros({3});

  // Like ATen, the scalar takes the tensor's dtype: 255 is -1 and 257 is 1 as
  // int8.
  op_fmod_scalar_out(self, 255, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({3}));
  op_fmod_scalar_out(self, 257, out);
  EXPECT_TENSOR_EQ(out, tf.zeros({3}));

  // 256 is 0 as int8, and 2^32 is 0 as int32.
  ET_EXPECT_KERNEL_FAILURE(context_, op_fmod_scalar_out(self, 256, out));
  TensorFactory<ScalarType::Byte> tf_byte;
  Tensor out_byte = tf_byte.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_fmod_scalar_out(tf_byte.make({2}, {5, 200}), 256, out_byte));
  TensorFactory<ScalarType::Int> tf_int;
  Tensor out_int = tf_int.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_fmod_scalar_out(
          tf_int.make({2}, {5, 6}), Scalar(int64_t{1} << 32), out_int));
}

TEST_F(OpFmodTest, ScalarSmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;
  std::vector<float> a(18);
  std::iota(a.begin(), a.end(), -8);
  Tensor self = tfFloat.make({18}, a);
  Scalar other = 3;
  Tensor out = tfFloat.zeros({18});
  Tensor out_expected = tfFloat.make(
      {18},
      {-2.,
       -1.,
       -0.,
       -2.,
       -1.,
       -0.,
       -2.,
       -1.,
       0.,
       1.,
       2.,
       0.,
       1.,
       2.,
       0.,
       1.,
       2.,
       0.});
  op_fmod_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
