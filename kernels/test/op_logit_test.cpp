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
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_logit_out(const Tensor& self, optional<double> eps, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::logit_outf(context, self, eps, out);
}

TEST(OpLogitOutTest, DtypeTest_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972246170043945,
       2.1972246170043945,
       2.1972246170043945,
       2.1972246170043945});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972245773362196,
       2.1972245773362196,
       2.1972245773362196,
       2.1972245773362196});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2},
      {2.1972243785858154,
       -2.1972246170043945,
       -2.1972246170043945,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2},
      {2.1972243785858154,
       -2.1972246170043945,
       -2.1972246170043945,
       2.1972243785858154});
  op_logit_out(self, eps, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpLogitOutTest, DtypeTest_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

TEST(OpLogitOutTest, DtypeTest_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::optional<double> eps = exec_aten::optional<double>(0.1);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(op_logit_out(self, eps, out));
}

// Common testing for logit operator
template <ScalarType DTYPE, ScalarType OUTPUT_DTYPE>
void test_integer_logit_out() {
  TensorFactory<DTYPE> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the logit operator.
  Tensor out = tf_out.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(
      op_logit_out(tf.make(sizes, /*data=*/{1, 2, 4, 8}), 0, out));
}

template <>
void test_integer_logit_out<ScalarType::Float, ScalarType::Float>() {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the logit operator.
  Tensor out = tf_out.zeros(sizes);

  // Check that it matches (or close to) the expected output.
  op_logit_out(tf.make(sizes, /*data=*/{.1, .2, .4, .8}), 0, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          sizes, /*data=*/{-2.197224, -1.386294, -0.405465, 1.3862943}));
}

// Common testing for logit operator
template <ScalarType DTYPE, ScalarType OUTPUT_DTYPE>
void test_integer_logit_out_eps_set() {
  TensorFactory<DTYPE> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the logit operator.
  Tensor out = tf_out.zeros(sizes);

  op_logit_out(tf.make(sizes, /*data=*/{1, 2, 4, 8}), 0.1, out);

  // Check that it matches (or close to) the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(sizes, /*data=*/{2.197224, 2.197224, 2.197224, 2.197224}));
}

TEST(OpLogitOutKernelTest, AllRealInputFloatOutputSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle this";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_integer_logit_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpLogitOutKernelTest, AllRealInputDoubleOutputSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle this";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_integer_logit_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
TEST(OpLogitOutKernelTest, AllRealInputFloatOutputSupportEpsSet) {
#define TEST_ENTRY(ctype, dtype) \
  test_integer_logit_out_eps_set<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpLogitOutKernelTest, AllRealInputDoubleOutputSupportEpsSet) {
#define TEST_ENTRY(ctype, dtype) \
  test_integer_logit_out_eps_set<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Mismatched shape tests.
TEST(OpLogitOutKernelTest, MismatchedShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Float> tf_out;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf_out.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(op_logit_out(a, 0, out));
}

// Unhandled output dtypes.
template <ScalarType OUTPUT_DTYPE>
void test_logit_invalid_output_dtype_dies() {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 5};

  Tensor in = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_logit_out(in, 0, out));
}

TEST(OpLogitOutKernelTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_logit_invalid_output_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpLogitOutKernelTest, SimpleGeneratedCase) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  Tensor expected_result = tf.make(
      {10, 10}, {2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154, 2.1972243785858154, 2.1972243785858154,
                 2.1972243785858154});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_logit_out(x, 0.1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpLogitOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9622091054916382,
       0.511866569519043,
       0.15690308809280396,
       0.7423648834228516,
       0.627659797668457,
       0.4892460107803345});
  Tensor expected_result = tf.make(
      {3, 2},
      {2.1972243785858154,
       0.04747522622346878,
       -1.6814535856246948,
       1.05829656124115,
       0.5221903324127197,
       -0.043022606521844864});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_logit_out(x, 0.1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpLogitOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9622091054916382,
       0.511866569519043,
       0.15690308809280396,
       0.7423648834228516,
       0.627659797668457,
       0.4892460107803345});
  Tensor expected_result = tf.make(
      {3, 2},
      {2.1972243785858154,
       0.04747522622346878,
       -1.6814535856246948,
       1.05829656124115,
       0.5221903324127197,
       -0.043022606521844864});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_logit_out(x, 0.1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpLogitOutKernelTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9622091054916382,
       0.511866569519043,
       0.15690308809280396,
       0.7423648834228516,
       0.627659797668457,
       0.4892460107803345});
  Tensor expected_result = tf.make(
      {3, 2},
      {2.1972243785858154,
       0.04747522622346878,
       -1.6814535856246948,
       1.05829656124115,
       0.5221903324127197,
       -0.043022606521844864});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_logit_out(x, 0.1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
