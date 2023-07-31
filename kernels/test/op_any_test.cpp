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
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& _any_all_out(const Tensor& input, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::any_outf(context, input, out);
}

TEST(OpAnyAllOutTest, DtypeTest_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyAllOutTest, DtypeTest_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({});
  ET_EXPECT_KERNEL_FAILURE(_any_all_out(self, out));
}

TEST(OpAnyAllOutTest, DtypeTest_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _any_all_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAnyOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Float> tff;
  const std::vector<int32_t> size{2, 2};

  Tensor in = tff.make(size, {0, 0, 1, 0});
  Tensor out = tff.ones(/*size=*/{1, 1});

  ET_EXPECT_KERNEL_FAILURE(_any_all_out(in, out));
}

template <ScalarType OUT_DTYPE>
void test_any_all_out_invalid_type() {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<OUT_DTYPE> tf_out;

  Tensor in = tf_float.make(
      {1, 4},
      {
          0,
          0,
          1,
          0,
      });
  Tensor out = tf_out.zeros(/*size=*/{0});

  ET_EXPECT_KERNEL_FAILURE(_any_all_out(in, out));
}

TEST(OpAnyOutTest, InvalidDtypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_any_all_out_invalid_type<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

template <ScalarType IN_DTYPE>
void test_any_all_out() {
  TensorFactory<IN_DTYPE> tf_in;
  TensorFactory<ScalarType::Bool> tf_bool;
  // clang-format off
  Tensor in = tf_in.make(
    {2, 4},
    {
      0, 1, 0, 1,
      1, 0, 1, 0
    });
  Tensor bool_false_in = tf_bool.make(
    {2, 4},
    {
      false, false, false, false,
      false, false, false, false,
    });
  Tensor bool_true_in = tf_bool.make(
    {2, 4},
    {
      true, true, true, true,
      true, true, true, true,
    });
  // clang-format on

  Tensor out = tf_bool.make({}, {false});

  _any_all_out(in, out);
  EXPECT_TENSOR_EQ(out, tf_bool.make({}, {true}));

  _any_all_out(bool_false_in, out);
  EXPECT_TENSOR_EQ(out, tf_bool.make({}, {false}));

  _any_all_out(bool_true_in, out);
  EXPECT_TENSOR_EQ(out, tf_bool.make({}, {true}));
}

TEST(OpAnyOutTest, AllRealInputTypePasses) {
#define TEST_ENTRY(ctype, dtype) test_any_all_out<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
