// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& logit_out(const Tensor& self, optional<double> eps, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::logit_outf(context, self, eps, out);
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
      logit_out(tf.make(sizes, /*data=*/{1, 2, 4, 8}), 0, out));
}

template <>
void test_integer_logit_out<ScalarType::Float, ScalarType::Float>() {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the logit operator.
  Tensor out = tf_out.zeros(sizes);

  // Check that it matches (or close to) the expected output.
  logit_out(tf.make(sizes, /*data=*/{.1, .2, .4, .8}), 0, out);
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

  logit_out(tf.make(sizes, /*data=*/{1, 2, 4, 8}), 0.1, out);

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

TEST(OpLogitOutKernelTest, UnhandledInputDtypeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle bool input dtype";
  }
  // logit_out() doesn't handle Bool as input.
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};
  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});

  // Destination for the logit
  Tensor out = tf_out.zeros(sizes);

  // Boolean tensor should cause an assertion and kill the
  // test process.
  ET_EXPECT_KERNEL_FAILURE(logit_out(a, 0, out));
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

  ET_EXPECT_KERNEL_FAILURE(logit_out(a, 0, out));
}

// Unhandled output dtypes.
template <ScalarType OUTPUT_DTYPE>
void test_logit_invalid_output_dtype_dies() {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 5};

  Tensor in = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(logit_out(in, 0, out));
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
  Tensor ret = logit_out(x, 0.1, out);
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
  Tensor ret = logit_out(x, 0.1, out);
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
  Tensor ret = logit_out(x, 0.1, out);
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
  Tensor ret = logit_out(x, 0.1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
