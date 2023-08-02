/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::MemoryFormat;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_full_like_out(
    const Tensor& self,
    const Scalar& fill_value,
    optional<MemoryFormat> memory_format,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::full_like_outf(
      context, self, fill_value, memory_format, out);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 1.0, 1.0, 1.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 1, 1, 1});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(true);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {2.0, 2.0, 2.0, 2.0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {2, 2, 2, 2});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(2);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.5, 0.5, 0.5, 0.5});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpFullLikeOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar fill_value = exec_aten::Scalar(0.5);
  exec_aten::optional<exec_aten::MemoryFormat> memory_format;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, true, true, true});
  op_full_like_out(self, fill_value, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

template <ScalarType DTYPE>
void test_full_like_out() {
  TensorFactory<DTYPE> tf;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor in = tf.zeros(sizes);
  Tensor out = tf.zeros(sizes);
  Scalar value = 42;
  MemoryFormat memory_format = MemoryFormat::Contiguous;

  // Check that it matches the expected output.
  op_full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{42, 42, 42, 42}));

  value = 1;
  op_full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.ones(sizes));
}

template <>
void test_full_like_out<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor in = tf.zeros(sizes);
  Tensor out = tf.zeros(sizes);
  Scalar value = true;
  MemoryFormat memory_format = MemoryFormat::Contiguous;

  // Check that it matches the expected output.
  op_full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{true, true, true, true}));

  value = false;
  op_full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.zeros(sizes));
}

TEST(OpFullLikeTest, AllRealOutputPasses) {
#define TEST_ENTRY(ctype, dtype) test_full_like_out<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

template <ScalarType DTYPE>
void test_full_like_out_mismatched_shape() {
  TensorFactory<DTYPE> tf;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor in = tf.zeros(/*sizes=*/{2, 2});
  Tensor out = tf.zeros(/*sizes=*/{4, 2});
  Scalar value = 42;
  MemoryFormat memory_format;

  ET_EXPECT_DEATH(op_full_like_out(in, value, memory_format, out), "");
}

TEST(OpFullLikeTest, MismatchedShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_full_like_out_mismatched_shape<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpFullLikeTest, SimpleGeneratedCase) {
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
      {10, 10},
      {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpFullLikeTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make({3, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpFullLikeTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make({3, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpFullLikeTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make({3, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
