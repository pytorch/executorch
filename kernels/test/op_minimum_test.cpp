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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_minimum_out(const Tensor& self, const Tensor& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::minimum_outf(context, self, other, out);
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_minimum_out(self, other, out));
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpMinimumOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, false, false, true});
  op_minimum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

// Common testing for minimum operator
template <ScalarType DTYPE>
void test_minimum_out_same_size() {
  TensorFactory<DTYPE> tf;
  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the minimum operator.
  Tensor out = tf.zeros(sizes);

  op_minimum_out(
      tf.make(sizes, /*data=*/{1, 2, 4, 8}),
      tf.make(sizes, /*data=*/{3, 0, 4, 9}),
      out);

  // Check that it matches to the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{1, 0, 4, 8}));
}

TEST(OpMinimumOutKernelTest, ByteTensors) {
  test_minimum_out_same_size<ScalarType::Byte>();
}

TEST(OpMinimumOutKernelTest, CharTensors) {
  test_minimum_out_same_size<ScalarType::Char>();
}

TEST(OpMinimumOutKernelTest, ShortTensors) {
  test_minimum_out_same_size<ScalarType::Short>();
}

TEST(OpMinimumOutKernelTest, IntTensors) {
  test_minimum_out_same_size<ScalarType::Int>();
}

TEST(OpMinimumOutKernelTest, LongTensors) {
  test_minimum_out_same_size<ScalarType::Long>();
}

TEST(OpMinimumOutKernelTest, FloatTensors) {
  test_minimum_out_same_size<ScalarType::Float>();
}

TEST(OpMinimumOutKernelTest, DoubleTensors) {
  test_minimum_out_same_size<ScalarType::Double>();
}

TEST(OpMinimumOutKernelTest, BothScalarTensors) {
  // Checks the case when both cases are scalar.
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes = {1, 1};
  Tensor out = tf.zeros(sizes);
  op_minimum_out(tf.make(sizes, {1.2}), tf.make(sizes, {3.5}), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {1.2}));
}

TEST(OpMinimumOutKernelTest, LeftScalarTensor) {
  // Checks the case where one of the tensor is a singleton tensor.

  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes_1 = {1, 1};
  const std::vector<int32_t> sizes_2 = {2, 2};
  Tensor out1 = tf.zeros(sizes_2);
  Tensor out2 = tf.zeros(sizes_2);

  auto a = tf.make(sizes_1, /*data=*/{1.0});
  auto b = tf.make(sizes_2, /*data=*/{3.5, -1.0, 0.0, 5.5});

  // Case 1 : First argument is singleton.
  op_minimum_out(a, b, out1);
  EXPECT_TENSOR_EQ(out1, tf.make(sizes_2, {1.0, -1.0, 0.0, 1.0}));

  // Case 2: Second argument is singleton
  op_minimum_out(b, a, out2);
  EXPECT_TENSOR_EQ(out2, tf.make(sizes_2, {1.0, -1.0, 0.0, 1.0}));
}

TEST(OpMinimumOutKernelTest, MismatchedInputShapesDies) {
  // First and second argument have different shape
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      op_minimum_out(tf.ones({2, 2}), tf.ones({3, 3}), out));
}

TEST(OpMinimumOutKernelTest, MismatchedOutputShapesDies) {
  // First and second argument have same shape, but output has different shape.
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({3, 3});

  ET_EXPECT_KERNEL_FAILURE(
      op_minimum_out(tf.ones({2, 2}), tf.ones({3, 3}), out));
}

TEST(OpMinimumOutKernelTest, MismatchedOutputShapeWithSingletonDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }
  // First argument is singleton but second and output has different shape.
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({4, 4});

  ET_EXPECT_KERNEL_FAILURE(
      op_minimum_out(tf.ones({1, 1}), tf.ones({3, 3}), out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(3, 2)
y = torch.rand(3, 2)
res = torch.minimum(x, y)
op = "op_minimum_out"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST(OpMinimumOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpMinimumOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpMinimumOutKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}
