#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/supported_features.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::MemoryFormat;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& full_like_out(
    const Tensor& self,
    const Scalar& fill_value,
    optional<MemoryFormat> memory_format,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::full_like_outf(
      context, self, fill_value, memory_format, out);
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
  full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{42, 42, 42, 42}));

  value = 1;
  full_like_out(in, value, memory_format, out);
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
  full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{true, true, true, true}));

  value = false;
  full_like_out(in, value, memory_format, out);
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

  ET_EXPECT_DEATH(full_like_out(in, value, memory_format, out), "");
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

template <ScalarType DTYPE>
void test_full_like_out_invalid_type() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::QUInt8> tf_qint;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor in = tf.zeros(sizes);
  Tensor out = tf_qint.zeros(sizes);
  Scalar value = 42;
  MemoryFormat memory_format;

  ET_EXPECT_DEATH(full_like_out(in, value, memory_format, out), "");
}

TEST(OpFullLikeTest, InvalidOutputDTypeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_full_like_out_invalid_type<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpFullLikeTest, InvalidFillValueDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Char> tf_char;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor in = tf_int.zeros(sizes);
  Tensor out = tf_int.zeros(sizes);
  Scalar value = 4.2;
  MemoryFormat memory_format;

  // float -> int conversion should die
  ET_EXPECT_DEATH(full_like_out(in, value, memory_format, out), "");

  // int -> char overflow conversion should die
  out = tf_char.zeros(sizes);
  value = 129;
  ET_EXPECT_DEATH(full_like_out(in, value, memory_format, out), "");
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
  Tensor ret = full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
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
  Tensor ret = full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
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
  Tensor ret = full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
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
  Tensor ret = full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
