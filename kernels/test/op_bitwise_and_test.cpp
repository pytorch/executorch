// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor&
_bitwise_and_scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::bitwise_and_outf(context, self, other, out);
}

Tensor&
_bitwise_and_tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::bitwise_and_outf(context, self, other, out);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 1.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 1, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.0, 2.0, 2.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 2, 2, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, false, false, true});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {0.0, 0.0, 0.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {0.0, 0.0, 0.0, 0.0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {0, 0, 0, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {0, 0, 0, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {0, 0, 0, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {0, 0, 0, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {0, 0, 0, 0});
  _bitwise_and_scalar_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndScalarOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Scalar other = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_scalar_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 2.0, 3.0, 4.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 2, 3, 4});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 0.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_bitwise_and_tensor_out(self, other, out));
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfFloat.zeros({2, 2});
  exec_aten::Tensor out_expected = tfFloat.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfDouble.zeros({2, 2});
  exec_aten::Tensor out_expected = tfDouble.make({2, 2}, {1.0, 0.0, 0.0, 1.0});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfByte.zeros({2, 2});
  exec_aten::Tensor out_expected = tfByte.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfChar.zeros({2, 2});
  exec_aten::Tensor out_expected = tfChar.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfShort.zeros({2, 2});
  exec_aten::Tensor out_expected = tfShort.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfInt.zeros({2, 2});
  exec_aten::Tensor out_expected = tfInt.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfLong.zeros({2, 2});
  exec_aten::Tensor out_expected = tfLong.make({2, 2}, {1, 0, 0, 1});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpBitwiseAndTensorOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor self = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor other = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor out = tfBool.zeros({2, 2});
  exec_aten::Tensor out_expected =
      tfBool.make({2, 2}, {true, false, false, true});
  _bitwise_and_tensor_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
