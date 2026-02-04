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

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpConjPhysicalOutTest : public OperatorTest {
 protected:
  Tensor& op_conj_physical_out(const Tensor& in, Tensor& out) {
    return torch::executor::aten::_conj_physical_outf(context_, in, out);
  }
};

TEST_F(OpConjPhysicalOutTest, ComplexFloatBasic) {
  TensorFactory<ScalarType::ComplexFloat> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Create input: (1+2i), (3+4i), (5-6i), (-7+8i)
  Tensor in = tf.make(
      sizes,
      {executorch::aten::complex<float>(1.0f, 2.0f),
       executorch::aten::complex<float>(3.0f, 4.0f),
       executorch::aten::complex<float>(5.0f, -6.0f),
       executorch::aten::complex<float>(-7.0f, 8.0f)});

  Tensor out = tf.zeros(sizes);

  op_conj_physical_out(in, out);

  // Expected: (1-2i), (3-4i), (5+6i), (-7-8i)
  Tensor expected = tf.make(
      sizes,
      {executorch::aten::complex<float>(1.0f, -2.0f),
       executorch::aten::complex<float>(3.0f, -4.0f),
       executorch::aten::complex<float>(5.0f, 6.0f),
       executorch::aten::complex<float>(-7.0f, -8.0f)});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpConjPhysicalOutTest, ComplexDoubleBasic) {
  TensorFactory<ScalarType::ComplexDouble> tf;

  const std::vector<int32_t> sizes = {3};

  Tensor in = tf.make(
      sizes,
      {executorch::aten::complex<double>(1.5, 2.5),
       executorch::aten::complex<double>(-3.5, 4.5),
       executorch::aten::complex<double>(0.0, -1.0)});

  Tensor out = tf.zeros(sizes);

  op_conj_physical_out(in, out);

  Tensor expected = tf.make(
      sizes,
      {executorch::aten::complex<double>(1.5, -2.5),
       executorch::aten::complex<double>(-3.5, -4.5),
       executorch::aten::complex<double>(0.0, 1.0)});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpConjPhysicalOutTest, RealPartOnly) {
  TensorFactory<ScalarType::ComplexFloat> tf;

  const std::vector<int32_t> sizes = {2};

  // When imaginary part is zero, conjugate negates the imaginary part (0 -> -0)
  // Both are mathematically equivalent, so we verify values directly
  Tensor in = tf.make(
      sizes,
      {executorch::aten::complex<float>(5.0f, 0.0f),
       executorch::aten::complex<float>(-3.0f, 0.0f)});

  Tensor out = tf.zeros(sizes);

  op_conj_physical_out(in, out);

  // Verify real parts are unchanged and imaginary parts are negated zeros
  const auto* out_data = out.const_data_ptr<executorch::aten::complex<float>>();
  EXPECT_EQ(out_data[0].real_, 5.0f);
  EXPECT_EQ(out_data[0].imag_, -0.0f);
  EXPECT_EQ(out_data[1].real_, -3.0f);
  EXPECT_EQ(out_data[1].imag_, -0.0f);
}

TEST_F(OpConjPhysicalOutTest, ImaginaryPartOnly) {
  TensorFactory<ScalarType::ComplexFloat> tf;

  const std::vector<int32_t> sizes = {2};

  Tensor in = tf.make(
      sizes,
      {executorch::aten::complex<float>(0.0f, 5.0f),
       executorch::aten::complex<float>(0.0f, -3.0f)});

  Tensor out = tf.zeros(sizes);

  op_conj_physical_out(in, out);

  Tensor expected = tf.make(
      sizes,
      {executorch::aten::complex<float>(0.0f, -5.0f),
       executorch::aten::complex<float>(0.0f, 3.0f)});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpConjPhysicalOutTest, EmptyTensor) {
  TensorFactory<ScalarType::ComplexFloat> tf;

  const std::vector<int32_t> sizes = {0};

  Tensor in = tf.make(sizes, {});
  Tensor out = tf.zeros(sizes);

  op_conj_physical_out(in, out);

  EXPECT_EQ(out.numel(), 0);
}

TEST_F(OpConjPhysicalOutTest, MismatchedDtypeDies) {
  TensorFactory<ScalarType::ComplexFloat> tf_in;
  TensorFactory<ScalarType::ComplexDouble> tf_out;

  const std::vector<int32_t> sizes = {2};

  Tensor in = tf_in.make(
      sizes,
      {executorch::aten::complex<float>(1.0f, 2.0f),
       executorch::aten::complex<float>(3.0f, 4.0f)});
  Tensor out = tf_out.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_conj_physical_out(in, out));
}
