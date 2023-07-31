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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& _transpose_copy_int_out(
    const Tensor& self,
    int64_t dim0,
    int64_t dim1,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::transpose_copy_outf(
      context, self, dim0, dim1, out);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfFloat.make({2, 2}, {1.3125, 3.5, 2.625, 4.875});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfDouble.make({2, 2}, {1.3125, 3.5, 2.625, 4.875});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 3, 2, 4});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 3, 2, 4});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 3, 2, 4});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 3, 2, 4});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 3, 2, 4});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(self, dim0, dim1, out));
}

TEST(OpTransposeCopyIntOutTest, DtypeTest_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  int64_t dim0 = 0;
  int64_t dim1 = 1;
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, false, false, true});
  _transpose_copy_int_out(self, dim0, dim1, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpTransposeIntCopyKernelTest, TwoDTranspose) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 3}, {
    // 2x3 data block
    0, 1, 2,
    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2};
  Tensor out = tf.zeros(new_sizes);

  _transpose_copy_int_out(t_int, 1, 0, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
    // 3x2 data block
    0, 3,
    1, 4,
    2, 5
  }));
  // clang-format on
}

TEST(OpTransposeIntCopyKernelTest, TwoDNegativeIndices) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 3}, {
    // 2x3 data block
    0, 1, 2,
    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2};
  Tensor out = tf.zeros(new_sizes);

  _transpose_copy_int_out(t_int, -1, -2, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
    // 3x2 data block
    0, 3,
    1, 4,
    2, 5
  }));
  // clang-format on
}

TEST(OpTransposeIntCopyKernelTest, TransposeNoDatachange) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 1, 3}, {
    // 2 1x3 data blocks
    0, 1, 2,

    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {2, 3, 1};
  Tensor out = tf.zeros(new_sizes);

  _transpose_copy_int_out(t_int, 1, 2, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
  // 2 3x1 data blocks
    0,
    1,
    2,

    3,
    4,
    5,
  }));
  // clang-format on
}

TEST(OpTransposeIntCopyKernelTest, ThreeDTranspose) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 2, 3}, {
    // 2 2x3 data blocks
    0, 1, 2,
    3, 4, 5,

    6, 7, 8,
    9, 10, 11
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2, 2};
  Tensor out = tf.zeros(new_sizes);

  _transpose_copy_int_out(t_int, 0, 2, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
  // 3 2x2 data blocks
    0, 6,
    3, 9,

    1, 7,
    4, 10,

    2,  8,
    5, 11
  }));
  // clang-format on
}

// transpose an out of bounds dim
TEST(OpTransposeIntCopyKernelTest, OutOfBoundDimDies) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{2, 3});
  Tensor out = tf.ones(/*sizes=*/{3, 2});

  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(a, 0, -3, out));
}

// transpose a 3d tensor into a 2d one
TEST(OpTransposeIntCopyKernelTest, MismatchedDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{4, 2, 3});
  Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(_transpose_copy_int_out(a, 0, 1, out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.randint(10, (2, 2, 3))
res = torch.transpose(x, 0, 2)
op = "_transpose_copy_int_out"
opt_extra_params = "0, 2,"
dtype = "ScalarType::Int"
check = "EXPECT_TENSOR_EQ" */

TEST(OpTransposeIntCopyKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
  Tensor expected = tf.make({3, 2, 2}, {4, 7, 0, 3, 9, 3, 3, 1, 3, 7, 9, 6});

  Tensor out =
      tf.zeros({3, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _transpose_copy_int_out(x, 0, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpTransposeIntCopyKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
  Tensor expected = tf.make({3, 2, 2}, {4, 7, 0, 3, 9, 3, 3, 1, 3, 7, 9, 6});

  Tensor out =
      tf.zeros({5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _transpose_copy_int_out(x, 0, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpTransposeIntCopyKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
  Tensor expected = tf.make({3, 2, 2}, {4, 7, 0, 3, 9, 3, 3, 1, 3, 7, 9, 6});

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  _transpose_copy_int_out(x, 0, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}
