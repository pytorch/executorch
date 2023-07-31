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
using exec_aten::IntArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& _constant_pad_nd_out(
    const Tensor& self,
    const IntArrayRef padding,
    const Scalar& value,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::constant_pad_nd_outf(
      context, self, padding, value, out);
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  exec_aten::Tensor out_expected = tfFloat.make(
      {3, 3}, {1.0, 1.3125, 2.625, 1.0, 3.5, 4.875, 1.0, 1.0, 1.0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  exec_aten::Tensor out_expected = tfFloat.make(
      {3, 3}, {2.0, 1.3125, 2.625, 2.0, 3.5, 4.875, 2.0, 2.0, 2.0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  exec_aten::Tensor out_expected = tfFloat.make(
      {3, 3}, {0.5, 1.3125, 2.625, 0.5, 3.5, 4.875, 0.5, 0.5, 0.5});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  exec_aten::Tensor out_expected = tfDouble.make(
      {3, 3}, {1.0, 1.3125, 2.625, 1.0, 3.5, 4.875, 1.0, 1.0, 1.0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  exec_aten::Tensor out_expected = tfDouble.make(
      {3, 3}, {2.0, 1.3125, 2.625, 2.0, 3.5, 4.875, 2.0, 2.0, 2.0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  exec_aten::Tensor out_expected = tfDouble.make(
      {3, 3}, {0.5, 1.3125, 2.625, 0.5, 3.5, 4.875, 0.5, 0.5, 0.5});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfByte.make({3, 3}, {1, 1, 2, 1, 3, 4, 1, 1, 1});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfByte.make({3, 3}, {2, 1, 2, 2, 3, 4, 2, 2, 2});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfByte.make({3, 3}, {0, 1, 2, 0, 3, 4, 0, 0, 0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfChar.make({3, 3}, {1, 1, 2, 1, 3, 4, 1, 1, 1});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfChar.make({3, 3}, {2, 1, 2, 2, 3, 4, 2, 2, 2});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfChar.make({3, 3}, {0, 1, 2, 0, 3, 4, 0, 0, 0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfShort.make({3, 3}, {1, 1, 2, 1, 3, 4, 1, 1, 1});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfShort.make({3, 3}, {2, 1, 2, 2, 3, 4, 2, 2, 2});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfShort.make({3, 3}, {0, 1, 2, 0, 3, 4, 0, 0, 0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfInt.make({3, 3}, {1, 1, 2, 1, 3, 4, 1, 1, 1});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfInt.make({3, 3}, {2, 1, 2, 2, 3, 4, 2, 2, 2});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfInt.make({3, 3}, {0, 1, 2, 0, 3, 4, 0, 0, 0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfLong.make({3, 3}, {1, 1, 2, 1, 3, 4, 1, 1, 1});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfLong.make({3, 3}, {2, 1, 2, 2, 3, 4, 2, 2, 2});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  exec_aten::Tensor out_expected =
      tfLong.make({3, 3}, {0, 1, 2, 0, 3, 4, 0, 0, 0});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  exec_aten::Tensor out_expected = tfBool.make(
      {3, 3}, {true, true, false, true, false, true, true, true, true});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  exec_aten::Tensor out_expected = tfBool.make(
      {3, 3}, {true, true, false, true, false, true, true, true, true});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({3, 3});
  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, pad, value, out));
}

TEST(OpConstantPadNdOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  ::std::vector<int64_t> pad_vec = {1, 0, 0, 1};
  exec_aten::ArrayRef<int64_t> pad =
      exec_aten::ArrayRef<int64_t>(pad_vec.data(), pad_vec.size());
  exec_aten::Scalar value = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({3, 3});
  exec_aten::Tensor out_expected = tfBool.make(
      {3, 3}, {true, true, false, true, false, true, true, true, true});
  _constant_pad_nd_out(self, pad, value, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

template <ScalarType DTYPE>
void test_constant_pad_nd_out_dim2() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 4, 4};
  const std::vector<int32_t> sizes_out = {2, 4, 6};
  const std::vector<int64_t> padding = {1, 1};

  // clang-format off
  Tensor self = tf.make(
      sizes,
      {
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      sizes_out,
      {
         7,  1,  2,  3,  4,  7,
         7,  5,  6,  7,  8,  7,
         7,  1,  2,  3,  4,  7,
         7,  5,  6,  7,  8,  7,

         7,  1,  2,  3,  4,  7,
         7,  5,  6,  7,  8,  7,
         7,  1,  2,  3,  4,  7,
         7,  5,  6,  7,  8,  7,
      });
  // clang-format on

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
  Tensor out = tf.zeros(sizes_out);

  // Valid input should give the expected output
  _constant_pad_nd_out(self, padding_ref, 7, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

template <ScalarType DTYPE>
void test_constant_pad_nd_out_dim1() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 4, 4};
  const std::vector<int32_t> sizes_out = {2, 6, 4};
  const std::vector<int64_t> padding = {0, 0, 2, 0};

  // clang-format off
  Tensor self = tf.make(
      sizes,
      {
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      sizes_out,
      {
         7,  7,  7,  7,
         7,  7,  7,  7,
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         7,  7,  7,  7,
         7,  7,  7,  7,
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
  Tensor out = tf.zeros(sizes_out);

  // Valid input should give the expected output
  _constant_pad_nd_out(self, padding_ref, 7, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

template <ScalarType DTYPE>
void test_constant_pad_nd_out_dim0() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 4, 4};
  const std::vector<int32_t> sizes_out = {3, 4, 4};
  const std::vector<int64_t> padding = {0, 0, 0, 0, 1, 0};

  // clang-format off
  Tensor self = tf.make(
      sizes,
      {
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      sizes_out,
      {
         7,  7,  7,  7,
         7,  7,  7,  7,
         7,  7,  7,  7,
         7,  7,  7,  7,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
  Tensor out = tf.zeros(sizes_out);

  // Valid input should give the expected output
  _constant_pad_nd_out(self, padding_ref, 7, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

template <ScalarType DTYPE>
void test_constant_pad_nd_out_dim12() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 4, 4};
  const std::vector<int32_t> sizes_out = {2, 6, 7};
  const std::vector<int64_t> padding = {2, 1, 0, 2};

  // clang-format off
  Tensor self = tf.make(
      sizes,
      {
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      sizes_out,
      {
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,

         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
      });
  // clang-format on

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
  Tensor out = tf.zeros(sizes_out);

  // Valid input should give the expected output
  _constant_pad_nd_out(self, padding_ref, 7, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

template <ScalarType DTYPE>
void test_constant_pad_nd_out_dim02() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 4, 4};
  const std::vector<int32_t> sizes_out = {3, 4, 7};
  const std::vector<int64_t> padding = {2, 1, 0, 0, 0, 1};

  // clang-format off
  Tensor self = tf.make(
      sizes,
      {
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      sizes_out,
      {
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,

         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,

         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
      });
  // clang-format on

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
  Tensor out = tf.zeros(sizes_out);

  // Valid input should give the expected output
  _constant_pad_nd_out(self, padding_ref, 7, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

template <ScalarType DTYPE>
void test_constant_pad_nd_out_dim012() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 4, 4};
  const std::vector<int32_t> sizes_out = {3, 5, 7};
  const std::vector<int64_t> padding = {2, 1, 1, 0, 0, 1};

  // clang-format off
  Tensor self = tf.make(
      sizes,
      {
         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,

         1,  2,  3,  4,
         5,  6,  7,  8,
         1,  2,  3,  4,
         5,  6,  7,  8,
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf.make(
      sizes_out,
      {
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,

         7,  7,  7,  7,  7,  7,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,
         7,  7,  1,  2,  3,  4,  7,
         7,  7,  5,  6,  7,  8,  7,

         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,
      });
  // clang-format on

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
  Tensor out = tf.zeros(sizes_out);

  // Valid input should give the expected output
  _constant_pad_nd_out(self, padding_ref, 7, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpConstantPadNDOutKernelTest, TestPadDim2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim2<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpConstantPadNDOutKernelTest, TestPadDim1) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim1<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpConstantPadNDOutKernelTest, TestPadDim0) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim0<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpConstantPadNDOutKernelTest, TestPadDim1And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim12<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpConstantPadNDOutKernelTest, TestPadDim0And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim02<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpConstantPadNDOutKernelTest, TestPadDim0And1And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim012<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpConstantPadNDOutKernelTest, DifferentInputOutputTypesFail) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> tf_out;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 6};
  const std::vector<int64_t> padding = {1, 1};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST(OpConstantPadNDOutKernelTest, OddNumberOfPaddingElementsFail) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {1, 1, 0};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST(OpConstantPadNDOutKernelTest, TooManyPaddingElementsFail) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {3, 2, 1, 1, 2, 1, 1, 0};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST(OpConstantPadNDOutKernelTest, IncorrectOutputShapeFail) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle reshape output";
  }

  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {1, 1};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(_constant_pad_nd_out(self, padding_ref, 0, out));
}
