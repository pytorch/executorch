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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& _scalar_tensor_out(const Scalar& s, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::scalar_tensor_outf(context, s, out);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfFloat.zeros({});
  exec_aten::Tensor out_expected = tfFloat.make({}, {1.0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfDouble.zeros({});
  exec_aten::Tensor out_expected = tfDouble.make({}, {1.0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {1});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfChar.zeros({});
  exec_aten::Tensor out_expected = tfChar.make({}, {1});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfShort.zeros({});
  exec_aten::Tensor out_expected = tfShort.make({}, {1});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfInt.zeros({});
  exec_aten::Tensor out_expected = tfInt.make({}, {1});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfLong.zeros({});
  exec_aten::Tensor out_expected = tfLong.make({}, {1});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_bool_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Scalar s = exec_aten::Scalar(true);
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfFloat.zeros({});
  exec_aten::Tensor out_expected = tfFloat.make({}, {2.0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfDouble.zeros({});
  exec_aten::Tensor out_expected = tfDouble.make({}, {2.0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {2});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfChar.zeros({});
  exec_aten::Tensor out_expected = tfChar.make({}, {2});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfShort.zeros({});
  exec_aten::Tensor out_expected = tfShort.make({}, {2});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfInt.zeros({});
  exec_aten::Tensor out_expected = tfInt.make({}, {2});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfLong.zeros({});
  exec_aten::Tensor out_expected = tfLong.make({}, {2});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_int32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Scalar s = exec_aten::Scalar(2);
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_float32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfFloat.zeros({});
  exec_aten::Tensor out_expected = tfFloat.make({}, {0.5});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_float64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Double>
      tfDouble;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfDouble.zeros({});
  exec_aten::Tensor out_expected = tfDouble.make({}, {0.5});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_uint8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfByte.zeros({});
  exec_aten::Tensor out_expected = tfByte.make({}, {0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_int8) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Char> tfChar;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfChar.zeros({});
  exec_aten::Tensor out_expected = tfChar.make({}, {0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_int16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Short> tfShort;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfShort.zeros({});
  exec_aten::Tensor out_expected = tfShort.make({}, {0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_int32) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Int> tfInt;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfInt.zeros({});
  exec_aten::Tensor out_expected = tfInt.make({}, {0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_int64) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Long> tfLong;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfLong.zeros({});
  exec_aten::Tensor out_expected = tfLong.make({}, {0});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpScalarTensorOutTest, DtypeTest_float32_bool) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Bool> tfBool;

  exec_aten::Scalar s = exec_aten::Scalar(0.5);
  exec_aten::Tensor out = tfBool.zeros({});
  exec_aten::Tensor out_expected = tfBool.make({}, {true});
  _scalar_tensor_out(s, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_0d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  _scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

#define GENERATE_TEST_0D(ctype, dtype)                      \
  TEST(OpScalarTensorOutKernelTest, dtype##TensorsDim0) {   \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(4); \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(8); \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(9); \
  }

ET_FORALL_REAL_TYPES(GENERATE_TEST_0D)

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_1d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{1};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  _scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_2d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{1, 1};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  _scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

template <typename CTYPE, ScalarType DTYPE>
void test_scalar_tensor_out_3d(CTYPE value) {
  TensorFactory<DTYPE> tf;

  std::vector<int32_t> sizes{1, 1, 1};
  Tensor expected = tf.make(sizes, /*data=*/{value});

  Tensor out = tf.ones(sizes);
  _scalar_tensor_out(value, out);

  EXPECT_TENSOR_EQ(out, expected);
}

#define GENERATE_TEST(ctype, dtype)                                    \
  TEST(OpScalarTensorOutKernelTest, dtype##Tensors) {                  \
    if (torch::executor::testing::SupportedFeatures::get()->is_aten) { \
      GTEST_SKIP() << "ATen kernel resizes output to shape {}";        \
    }                                                                  \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(7);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(7);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(7);            \
  }

ET_FORALL_REAL_TYPES(GENERATE_TEST)

TEST(OpScalarTensorOutKernelTest, InvalidOutShapeFails) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel will reshape output";
  }

  TensorFactory<ScalarType::Int> tf;
  std::vector<int32_t> sizes{1, 2, 1};

  Tensor out = tf.ones(sizes);
  ET_EXPECT_KERNEL_FAILURE(_scalar_tensor_out(7, out));
}
