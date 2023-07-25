// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& _embedding_out(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::embedding_outf(
      context, weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2, 2}, {1.3125, 2.625, 3.5, 4.875, 3.5, 4.875, 1.3125, 2.625});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  exec_aten::Tensor out_expected = tfFloat.make(
      {2, 2, 2}, {1.3125, 2.625, 3.5, 4.875, 3.5, 4.875, 1.3125, 2.625});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfFloat.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2, 2}, {1.3125, 2.625, 3.5, 4.875, 3.5, 4.875, 1.3125, 2.625});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  exec_aten::Tensor out_expected = tfDouble.make(
      {2, 2, 2}, {1.3125, 2.625, 3.5, 4.875, 3.5, 4.875, 1.3125, 2.625});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_float64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfDouble.make({2, 2}, {1.3125, 2.625, 3.5, 4.875});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfByte.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfByte.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_uint8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor weight = tfByte.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfChar.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfChar.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int8_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfChar.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfShort.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfShort.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int16_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfShort.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfInt.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfInt.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int32_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfInt.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfLong.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  exec_aten::Tensor out_expected =
      tfLong.make({2, 2, 2}, {1, 2, 3, 4, 3, 4, 1, 2});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_int64_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfLong.make({2, 2}, {1, 2, 3, 4});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfFloat.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_float64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfDouble.make({2, 2}, {0.0, 1.0, 1.0, 0.0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_uint8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfByte.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int8_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfChar.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int16_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfShort.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfInt.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  exec_aten::Tensor out_expected = tfBool.make(
      {2, 2, 2}, {true, false, false, true, false, true, true, false});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_int64_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfLong.make({2, 2}, {0, 1, 1, 0});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  exec_aten::Tensor out_expected = tfBool.make(
      {2, 2, 2}, {true, false, false, true, false, true, true, false});
  _embedding_out(weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfFloat.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfDouble.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfByte.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfChar.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfShort.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfInt.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfLong.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, DtypeTest_bool_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Tensor weight = tfBool.make({2, 2}, {true, false, false, true});
  exec_aten::Tensor indices = tfBool.make({2, 2}, {false, true, true, false});
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;
  exec_aten::Tensor out = tfBool.zeros({2, 2, 2});
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight, indices, padding_idx, scale_grad_by_freq, sparse, out));
}

TEST(OpEmbeddingOutTest, Smoke) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {2, 2},
    {
      1., 2.,
      0.5, 0.6,
    });
  // clang-format on
  Tensor out = tff.zeros({1, 2});
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({1}, {1});
  // clang-format on
  Tensor actual = _embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);
  // Embedding takes the ith entry in `weight` for i in `indices`. So out =
  // weight.index_select(indices.reshape(-1)), in this test, out = weight[1]
  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, tff.make({1, 2}, {0.5, 0.6}));
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
template <class CTYPE, exec_aten::ScalarType DTYPE>
void test_dtype() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor weight = tf.make(
    {3, 2},
    {
      1, 2,
      3, 4,
      5, 6,
    });
  Tensor indices = tfl.make(
    {1, 2},
    {0, 2}
  );
  // clang-format on
  Tensor out = tf.zeros({1, 2, 2});
  Tensor actual = _embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);

  Tensor expected = tf.make({1, 2, 2}, {1, 2, 5, 6});

  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpEmbeddingOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST(OpEmbeddingOutTest, IndicesMultiDims) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({1, 2, 3, 2});
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({1, 2, 3}, {1, 0, 2, 3, 4, 0});
  // clang-format on
  Tensor actual = _embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);
  // clang-format off
  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, tff.make({1, 2, 3, 2}, {
      0.5, 0.6, // weight[1]
      1., 2.,   // weight[0]
      0.1, 0.2, // weight[2]
      3., 4.,   // weight[3]
      5., 6.,   // weight[4]
      1., 2.,   // weight[0]
  }));
  // clang-format on
}

TEST(OpEmbeddingOutTest, WeightWrongDimensionsDies) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {2, 2, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
    });
  // clang-format on
  Tensor out = tff.zeros({2, 2, 2});
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({2, 2}, {1, 0, 2, 3});
  // clang-format on
  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out));
}

TEST(OpEmbeddingOutTest, WrongOutShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong out shape";
  }
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  auto wrong_outs = {
      tff.zeros({4, 3}), tff.zeros({4, 2}), tff.zeros({4, 2, 2})};

  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor indices = tfl.make({2, 2}, {1, 0, 2, 3});

  for (auto wrong_out: wrong_outs) {
    // clang-format on
    ET_EXPECT_KERNEL_FAILURE(_embedding_out(
        weight,
        indices,
        /*padding_idx=*/0,
        /*scale_grad_by_freq=*/false,
        /*sparse=*/false,
        wrong_out));
  }
}

TEST(OpEmbeddingOutTest, UnmatchedOutTypeDie) {
  TensorFactory<ScalarType::Float> tff;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });

  Tensor wrong_out = tfl.zeros({2, 2, 2});
  Tensor indices = tfl.make({2, 2}, {1, 0, 2, 3});
  // clang-format on

  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      wrong_out));
}

TEST(OpEmbeddingOutTest, OutOfBoundIndicesDies) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({2, 2, 2});
  TensorFactory<ScalarType::Long> tfl;

  Tensor neg_indices = tfl.make({2, 2}, {-1, 0, 2, 4});
  Tensor overflow_indices = tfl.make({2, 2}, {1, 0, 2, 8});

  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight,
      neg_indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out));

  ET_EXPECT_KERNEL_FAILURE(_embedding_out(
      weight,
      overflow_indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out));
}

TEST(OpEmbeddingOutTest, EmptyWeightSupported) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 0},
    {});
  // clang-format on
  Tensor out = tff.ones({2, 2, 0});
  TensorFactory<ScalarType::Long> tfl;

  Tensor indices = tfl.make({2, 2}, {2, 0, 2, 4});

  Tensor actual = _embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);

  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(actual, tff.zeros({2, 2, 0}));
}

TEST(OpEmbeddingOutTest, ZeroDimIndicesSupported) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({2});
  TensorFactory<ScalarType::Long> tfl;

  Tensor indices = tfl.make({}, {3});

  // clang-format off
  Tensor expected = tff.make(
    {2},
    {3., 4.,}
  );
  // clang-format on

  Tensor actual = _embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);

  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpEmbeddingOutTest, EmptyDimIndicesSupported) {
  TensorFactory<ScalarType::Float> tff;
  // clang-format off
  Tensor weight = tff.make(
    {5, 2},
    {
      1., 2.,
      0.5, 0.6,
      0.1, 0.2,
      3., 4.,
      5., 6.,
    });
  // clang-format on
  Tensor out = tff.zeros({3, 0, 2});
  TensorFactory<ScalarType::Long> tfl;

  Tensor indices = tfl.make({3, 0}, {});

  // clang-format off
  Tensor expected = tff.make(
    {3, 0, 2},
    {}
  );
  // clang-format on

  Tensor actual = _embedding_out(
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      out);

  EXPECT_TENSOR_EQ(actual, out);
  EXPECT_TENSOR_EQ(out, expected);
}

/* %python
import torch
torch.manual_seed(0)
weight = torch.rand(10, 3)
indices = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
padding = 0
scale = False
sparse = False
expected = torch.nn.functional.embedding(
  indices, weight, padding_idx=padding, scale_grad_by_freq=scale, sparse=sparse
)
embedding_template = f"""
  {declare_tensor_factory("ScalarType::Float", "tf_weight")}
  {declare_tensor_factory("ScalarType::Long", "tf_indices")}

  {declare_tensor_make_t("weight", "tf_weight")}
  {declare_tensor_make_t("indices", "tf_indices")}
  {declare_tensor_make_t("expected", "tf_weight")}
  {declare_tensor_zeros("out_shape, dynamism", "tf_weight", "out")}

  _embedding_out(weight, indices, $padding$, $scale$, $sparse$, out);
  EXPECT_TENSOR_CLOSE(out, expected);""" */

void test_dynamic_shape(
    const std::vector<int32_t>& out_shape,
    enum torch::executor::TensorShapeDynamism dynamism) {
  /* %python
  %rewrite(embedding_template) */

  TensorFactory<ScalarType::Float> tf_weight;
  TensorFactory<ScalarType::Long> tf_indices;

  Tensor weight = tf_weight.make(
      {10, 3},
      {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
       0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
       0.9151939749717712,   0.39709991216659546, 0.8741558790206909,
       0.41940832138061523,  0.5529070496559143,  0.9527381062507629,
       0.036164820194244385, 0.1852310299873352,  0.37341737747192383});
  Tensor indices = tf_indices.make({2, 4}, {1, 2, 4, 5, 4, 3, 2, 9});
  Tensor expected = tf_weight.make(
      {2, 4, 3},
      {0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.518521785736084,    0.6976675987243652,  0.800011396408081,
       0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
       0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
       0.4900934100151062,   0.8964447379112244,  0.455627977848053,
       0.036164820194244385, 0.1852310299873352,  0.37341737747192383});
  Tensor out = tf_weight.zeros(out_shape, dynamism);

  _embedding_out(weight, indices, 0, false, false, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpEmbeddingOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 4, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpEmbeddingOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpEmbeddingOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
