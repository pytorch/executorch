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
using torch::executor::testing::TensorFactory;

Tensor& _fill_scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::fill_outf(context, self, other, out);
}

Tensor& _fill_tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::fill_outf(context, self, other, out);
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, value, out));
}

TEST(OpFillScalarOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_scalar_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfFloat.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfDouble.make({}, {1.0});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfByte.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfChar.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfShort.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfInt.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfLong.make({}, {1});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, value, out));
}

TEST(OpFillTensorOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor value = tfBool.make({}, {true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  _fill_tensor_out(self, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

template <ScalarType DTYPE>
void test_fill_scalar_out(std::vector<int32_t>&& sizes) {
  TensorFactory<DTYPE> tf;

  // Before: `out` consists of 0s.
  Tensor self = tf.zeros(sizes);
  Tensor out = tf.zeros(sizes);

  // After: `out` consists of 1s.
  Scalar other = 1;
  if (DTYPE == ScalarType::Bool) {
    other = false;
  }
  _fill_scalar_out(self, other, out);

  Tensor exp_out = tf.full(sizes, 1);
  if (DTYPE == ScalarType::Bool) {
    exp_out = tf.full(sizes, false);
  }

  // Check `out` matches expected output.
  EXPECT_TENSOR_EQ(out, exp_out);
}

template <ScalarType DTYPE>
void test_fill_tensor_out(std::vector<int32_t>&& sizes) {
  TensorFactory<DTYPE> tf;

  // Before: `out` consists of 0s.
  Tensor self = tf.zeros(sizes);
  Tensor out = tf.zeros(sizes);

  // After: `out` consists of 1s.
  Tensor other = tf.ones({});
  _fill_tensor_out(self, other, out);

  Tensor exp_out = tf.full(sizes, 1);

  // Check `out` matches expected output.
  EXPECT_TENSOR_EQ(out, exp_out);
}

// A macro for defining tests for both scalar and tensor variants of
// `fill_out`. Here the `self` and `out` tensors will be created according
// to the sizes provided, while the scalar/tensor will be a singleton.
#define TEST_FILL_OUT(FN, DTYPE)    \
  FN<ScalarType::DTYPE>({});        \
  FN<ScalarType::DTYPE>({1});       \
  FN<ScalarType::DTYPE>({1, 1, 1}); \
  FN<ScalarType::DTYPE>({2, 0, 4}); \
  FN<ScalarType::DTYPE>({2, 3, 4});

// Create input support tests for scalar variant.
#define GENERATE_SCALAR_INPUT_SUPPORT_TEST(_, DTYPE) \
  TEST(OpFillTest, DTYPE##ScalarInputSupport) {      \
    TEST_FILL_OUT(test_fill_scalar_out, DTYPE);      \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_SCALAR_INPUT_SUPPORT_TEST)

// Create input support tests for tensor variant.
#define GENERATE_TENSOR_INPUT_SUPPORT_TEST(_, DTYPE) \
  TEST(OpFillTest, DTYPE##TensorInputSupport) {      \
    TEST_FILL_OUT(test_fill_tensor_out, DTYPE);      \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_TENSOR_INPUT_SUPPORT_TEST)

TEST(OpFillTest, MismatchedOtherPropertiesDies) {
  TensorFactory<ScalarType::Int> tf;

  // `self` and `out` have different shapes but same dtype.
  Tensor self = tf.zeros({1});
  Tensor out = tf.zeros({1});

  // Create `other` tensors with incompatible shapes (`dim()` >=1) and/or
  // elements (`numel()` > 1).

  Tensor other1 = tf.zeros({1});
  EXPECT_EQ(other1.dim(), 1);
  EXPECT_EQ(other1.numel(), 1);

  Tensor other2 = tf.zeros({2});
  EXPECT_EQ(other2.dim(), 1);
  EXPECT_EQ(other2.numel(), 2);

  Tensor other3 = tf.zeros({3, 3});
  EXPECT_EQ(other3.dim(), 2);
  EXPECT_EQ(other3.numel(), 9);

  // Assert `other` tensors with incompatible properties fails.
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, other1, out));
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, other2, out));
  ET_EXPECT_KERNEL_FAILURE(_fill_tensor_out(self, other3, out));
}

TEST(OpFillTest, MismatchedOutputShapesDies) {
  // Skip ATen test since it supports `self` and `out` having different shapes.
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }

  TensorFactory<ScalarType::Int> tf;

  // `self` and `out` have different shapes but same dtype.
  Tensor self = tf.zeros({1});
  Tensor out = tf.zeros({2, 2});

  // Assert `out` can't be filled due to incompatible shapes.
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, 0, out));
}

TEST(OpFillTest, MismatchedOutputDtypeDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Float> tf_float;

  // `self` and `out` have different dtypes but same shape.
  Tensor self = tf_byte.zeros({2, 2});
  Tensor out = tf_float.ones({2, 2});

  // Assert `out` can't be filled due to incompatible dtype.
  ET_EXPECT_KERNEL_FAILURE(_fill_scalar_out(self, 0.0, out));
}
