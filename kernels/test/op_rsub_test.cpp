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
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& _rsub_scalar_out(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::rsub_outf(context, self, other, alpha, out);
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-0.625, -3.25, -5.0, -7.75});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {1.34375, 0.6875, 0.25, -0.4375});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-2.125, -4.75, -6.5, -9.25});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-0.15625, -0.8125, -1.25, -1.9375});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float32_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {-0.625, -3.25, -5.0, -7.75});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {1.34375, 0.6875, 0.25, -0.4375});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {-2.125, -4.75, -6.5, -9.25});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {-0.15625, -0.8125, -1.25, -1.9375});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_float64_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 254, 252, 250});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-1.5, -3.5, -5.5, -7.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {0.0, -0.5, -1.0, -1.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_uint8_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, -2, -4, -6});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-1.5, -3.5, -5.5, -7.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {0.0, -0.5, -1.0, -1.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int8_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, -2, -4, -6});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-1.5, -3.5, -5.5, -7.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {0.0, -0.5, -1.0, -1.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int16_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, -2, -4, -6});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-1.5, -3.5, -5.5, -7.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {0.0, -0.5, -1.0, -1.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int32_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, -2, -4, -6});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {-1.5, -3.5, -5.5, -7.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {0.0, -0.5, -1.0, -1.5});
  _rsub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_int64_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

TEST(OpRsubScalarOutTest, DtypeTest_bool_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Scalar alpha = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(self, other, alpha, out));
}

// Common testing for substraction of scalar for integer Tensor.
template <ScalarType DTYPE>
void test_integer_rsub_scalar_out() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the rsub.
  Tensor out = tf.zeros(sizes);

  // Performs substraction of tensor from scalar.
  _rsub_scalar_out(
      tf.make(sizes, /*data=*/{1, 2, 4, 5}),
      10,
      /*alpha=*/2,
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{8, 6, 2, 0}));
}

TEST(OpRSubScalarOutKernelTest, ByteTensors) {
  test_integer_rsub_scalar_out<ScalarType::Byte>();
}

TEST(OpSubOutKernelTest, CharTensors) {
  test_integer_rsub_scalar_out<ScalarType::Char>();
}

TEST(OpSubOutKernelTest, ShortTensors) {
  test_integer_rsub_scalar_out<ScalarType::Short>();
}

TEST(OpSubOutKernelTest, IntTensors) {
  test_integer_rsub_scalar_out<ScalarType::Int>();
}

TEST(OpSubOutKernelTest, LongTensors) {
  test_integer_rsub_scalar_out<ScalarType::Long>();
}

TEST(OpRSubScalarOutKernelTest, IntTensorFloatAlphaDies) {
  // rsub_scalar_out() doesn't handle floating alpha for intergal inputs
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the op.
  Tensor out = tf.zeros(sizes);

  // Subtraction operation on integral tensor with floating alpha
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      _rsub_scalar_out(tf.ones(sizes), 0, /*alpha=*/.7, out));
}

// Common testing for substraction between floating point tensor and scalar.
template <ScalarType DTYPE>
void test_floating_point_rsub_scalar_out() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the rsub.
  Tensor out = tf.zeros(sizes);

  // Performs substraction of tensor from scalar.
  _rsub_scalar_out(
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}),
      1.1,
      /*alpha=*/1,
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{0.0, -1.1, -3.3, -7.7}));
}

TEST(OpRSubScalarOutKernelTest, FloatTensors) {
  test_floating_point_rsub_scalar_out<ScalarType::Float>();
}

TEST(OpRSubScalarOutKernelTest, DoubleTensors) {
  test_floating_point_rsub_scalar_out<ScalarType::Double>();
}

TEST(OpRSubScalarOutKernelTest, UnhandledDtypeDies) {
  // rsub_scalar_out() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Subtrahend
  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});

  // Destination for the subtraction.
  Tensor out = tf.zeros(sizes);

  // Subtraction operation on boolean tensor should cause an assertion and
  // kill the test process.
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(a, false, /*alpha=*/0, out));
}

// The output tensor may not have a dtype different from the input even if it
// has the same shape.
TEST(OpRSubOutKernelTest, MismatchedOutputDtypeDies) {
  // Two different dtypes. This test uses two types with the same size to
  // demonstrate that the ScalarType itself matters, not the size of the
  // tensor elements.
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;

  const std::vector<int32_t> sizes = {2, 2};

  // Minuend and subtrahend of the same dtype.
  Tensor a = tf_byte.ones(sizes);

  // Destination with a dtype different from the inputs.
  Tensor out = tf_char.zeros(sizes);

  // Performing substraction of scalar from tesnor and write into a mismatched
  // output should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(a, 1, /*alpha=*/0, out));
}

// Mismatched shape tests.

TEST(OpRSubScalarOutKernelTest, MismatchedOutputShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle output shapes";
  }

  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.ones(sizes);

  // Destination with a different shape.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Performing substraction of scalar from tensor into a mismatched output
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(_rsub_scalar_out(a, 1, /*alpha=*/0, out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 3)
other = 10
alpha = 2
res = other - alpha * x
op = "_rsub_scalar_out"
opt_setup_params = f"""
  Scalar other = {other};
  Scalar alpha = {alpha};
"""
opt_extra_params = "other, alpha,"
out_args = "out_shape, dynamism"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_CLOSE" */

void test_dynamic_shape(
    const std::vector<int32_t>& out_shape,
    enum torch::executor::TensorShapeDynamism dynamism) {
  /* %python
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 3},
      {9.007486343383789,
       8.463556289672852,
       9.823044776916504,
       9.735939025878906,
       9.385154724121094,
       8.731842994689941});

  Scalar other = 10;
  Scalar alpha = 2;

  Tensor out = tf.zeros(out_shape, dynamism);
  _rsub_scalar_out(x, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpRSubScalarOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpRSubScalarOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpRSubScalarOutKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
