/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <cmath>
#include <limits>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorShapeDynamism;
using torch::executor::testing::TensorFactory;

class OpComplexOutTest : public OperatorTest {
 protected:
  Tensor& op_complex_out(const Tensor& real, const Tensor& imag, Tensor& out) {
    return torch::executor::aten::complex_outf(context_, real, imag, out);
  }

  template <ScalarType REAL, ScalarType COMPLEX>
  void test_dtype() {
    TensorFactory<REAL> tf;
    TensorFactory<COMPLEX> cf;
    using C = typename TensorFactory<COMPLEX>::true_ctype;
    Tensor real = tf.make({2, 2}, {1, -2, 0, 4});
    Tensor imag = tf.make({2, 2}, {-5, 0, 7, -8});
    Tensor out = cf.full({2, 2}, C(0, 0));
    EXPECT_EQ(&op_complex_out(real, imag, out), &out);
    EXPECT_TENSOR_EQ(
        out, cf.make({2, 2}, {C(1, -5), C(-2, 0), C(0, 7), C(4, -8)}));
  }

  template <ScalarType DTYPE>
  void test_unsupported_input() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::ComplexFloat> cf;
    Tensor input = tf.zeros({2});
    Tensor out = cf.zeros({2});
    ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(input, input, out));
  }
};

TEST_F(OpComplexOutTest, Half) {
  test_dtype<ScalarType::Half, ScalarType::ComplexHalf>();
}

TEST_F(OpComplexOutTest, Float) {
  test_dtype<ScalarType::Float, ScalarType::ComplexFloat>();
}

TEST_F(OpComplexOutTest, Double) {
  test_dtype<ScalarType::Double, ScalarType::ComplexDouble>();
}

TEST_F(OpComplexOutTest, SpecialValues) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor real = tf.make({4}, {0.0f, -0.0f, inf, nan});
  Tensor imag = tf.make({4}, {-0.0f, 0.0f, nan, -inf});
  Tensor out = cf.zeros({4});
  op_complex_out(real, imag, out);
  const auto* data = out.const_data_ptr<executorch::aten::complex<float>>();
  EXPECT_EQ(data[0].real(), 0.0f);
  EXPECT_FALSE(std::signbit(data[0].real()));
  EXPECT_EQ(data[0].imag(), 0.0f);
  EXPECT_TRUE(std::signbit(data[0].imag()));
  EXPECT_EQ(data[1].real(), 0.0f);
  EXPECT_TRUE(std::signbit(data[1].real()));
  EXPECT_EQ(data[1].imag(), 0.0f);
  EXPECT_FALSE(std::signbit(data[1].imag()));
  EXPECT_EQ(data[2].real(), inf);
  EXPECT_TRUE(std::isnan(data[2].imag()));
  EXPECT_TRUE(std::isnan(data[3].real()));
  EXPECT_EQ(data[3].imag(), -inf);
}

TEST_F(OpComplexOutTest, BroadcastBothInputs) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  using C = executorch::aten::complex<float>;
  Tensor real = tf.make({2, 1}, {1, 2});
  Tensor imag = tf.make({1, 3}, {3, 4, 5});
  Tensor out = cf.zeros({2, 3});
  op_complex_out(real, imag, out);
  EXPECT_TENSOR_EQ(
      out,
      cf.make({2, 3}, {C(1, 3), C(1, 4), C(1, 5), C(2, 3), C(2, 4), C(2, 5)}));
}

TEST_F(OpComplexOutTest, BroadcastDifferentRanks) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  using C = executorch::aten::complex<float>;
  Tensor real = tf.make({2}, {1, 2});
  Tensor imag = tf.make({2, 1}, {3, 4});
  Tensor out = cf.zeros({2, 2});
  op_complex_out(real, imag, out);
  EXPECT_TENSOR_EQ(out, cf.make({2, 2}, {C(1, 3), C(2, 3), C(1, 4), C(2, 4)}));
}

TEST_F(OpComplexOutTest, Scalars) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  Tensor real = tf.make({}, {1});
  Tensor imag = tf.make({}, {-2});
  Tensor out = cf.zeros({});
  op_complex_out(real, imag, out);
  EXPECT_TENSOR_EQ(out, cf.make({}, {executorch::aten::complex<float>(1, -2)}));
}

TEST_F(OpComplexOutTest, BroadcastScalar) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  using C = executorch::aten::complex<float>;
  Tensor real = tf.make({}, {1});
  Tensor imag = tf.make({2}, {2, 3});
  Tensor out = cf.zeros({2});
  op_complex_out(real, imag, out);
  EXPECT_TENSOR_EQ(out, cf.make({2}, {C(1, 2), C(1, 3)}));
}

TEST_F(OpComplexOutTest, EmptyBroadcast) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  Tensor real = tf.zeros({0, 3});
  Tensor imag = tf.ones({1, 3});
  Tensor out = cf.zeros({0, 3});
  op_complex_out(real, imag, out);
  EXPECT_TENSOR_EQ(out, cf.zeros({0, 3}));
}

TEST_F(OpComplexOutTest, DynamicOutput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  using C = executorch::aten::complex<float>;
  Tensor real = tf.make({2, 1}, {1, 2});
  Tensor imag = tf.make({2}, {3, 4});
  Tensor out = cf.zeros({4, 4}, TensorShapeDynamism::DYNAMIC_BOUND);
  op_complex_out(real, imag, out);
  EXPECT_TENSOR_EQ(out, cf.make({2, 2}, {C(1, 3), C(1, 4), C(2, 3), C(2, 4)}));
}

TEST_F(OpComplexOutTest, MismatchedInputs) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> df;
  TensorFactory<ScalarType::ComplexFloat> cf;
  Tensor real = tf.ones({2});
  Tensor imag = df.ones({2});
  Tensor out = cf.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(real, imag, out));
}

TEST_F(OpComplexOutTest, WrongComplexOutputType) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexDouble> cf;
  Tensor input = tf.ones({2});
  Tensor out = cf.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(input, input, out));
}

TEST_F(OpComplexOutTest, RealOutputType) {
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.ones({2});
  Tensor out = tf.zeros({2});
  ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(input, input, out));
}

TEST_F(OpComplexOutTest, BFloat16Input) {
  test_unsupported_input<ScalarType::BFloat16>();
}

TEST_F(OpComplexOutTest, IntegerInput) {
  test_unsupported_input<ScalarType::Int>();
}

TEST_F(OpComplexOutTest, BoolInput) {
  test_unsupported_input<ScalarType::Bool>();
}

TEST_F(OpComplexOutTest, ComplexInput) {
  test_unsupported_input<ScalarType::ComplexFloat>();
}

TEST_F(OpComplexOutTest, IncompatibleShapes) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  Tensor real = tf.ones({2});
  Tensor imag = tf.ones({3});
  Tensor out = cf.zeros({3});
  ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(real, imag, out));
}

#ifndef USE_ATEN_LIB
TEST_F(OpComplexOutTest, OutputCapacityTooSmall) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  Tensor input = tf.ones({2, 3});
  Tensor out = cf.zeros({1, 1}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(input, input, out));
}

TEST_F(OpComplexOutTest, StaticOutputWrongShape) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  Tensor input = tf.ones({2, 3});
  Tensor out = cf.zeros({3, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(input, input, out));
}

TEST_F(OpComplexOutTest, NonDefaultDimOrder) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::ComplexFloat> cf;
  for (int index = 0; index < 3; ++index) {
    context_ = {};
    Tensor real = tf.ones({2, 3});
    Tensor imag = tf.ones({2, 3});
    Tensor out = cf.zeros({2, 3});
    if (index == 0) {
      real = tf.make_with_dimorder({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 0});
    } else if (index == 1) {
      imag = tf.make_with_dimorder({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 0});
    } else {
      out = cf.make_with_dimorder(
          {2, 3}, std::vector<executorch::aten::complex<float>>(6), {1, 0});
    }
    ET_EXPECT_KERNEL_FAILURE(context_, op_complex_out(real, imag, out));
  }
}
#endif
