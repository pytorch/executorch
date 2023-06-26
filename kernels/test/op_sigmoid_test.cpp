// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& _sigmoid_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::sigmoid_outf(context, self, out);
}

// Common testing for sigmoid operator
template <ScalarType DTYPE, ScalarType OUTPUT_DTYPE>
void test_integer_sigmoid_out() {
  TensorFactory<DTYPE> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the sigmoid operator.
  Tensor out = tf_out.zeros(sizes);

  _sigmoid_out(tf.make(sizes, /*data=*/{1, 2, 4, 8}), out);

  // Check that it matches (or close to) the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(sizes, /*data=*/{0.731059, 0.880797, 0.982014, 0.999665}));
}

TEST(OpSigmoidOutKernelTest, AllRealInputFloatOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_integer_sigmoid_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSigmoidOutKernelTest, AllRealInputDoubleOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_integer_sigmoid_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSigmoidOutKernelTest, UnhandledInputDtypeDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "aten can handle bool as input";
  }
  // sigmdoid_out() doesn't handle Bool as input.
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};
  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});

  // Destination for the sigmoid
  Tensor out = tf_out.zeros(sizes);

  // Boolean tensor should cause an assertion and kill the
  // test process.
  ET_EXPECT_KERNEL_FAILURE(_sigmoid_out(a, out));
}

// Mismatched shape tests.
TEST(OpSigmoidOutKernelTest, MismatchedShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Float> tf_out;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf_out.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(_sigmoid_out(a, out));
}

// Unhandled output dtypes.
template <ScalarType OUTPUT_DTYPE>
void test_sigmoid_invalid_output_dtype_dies() {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 5};

  Tensor in = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(_sigmoid_out(in, out));
}

TEST(OpTanhOutKernelTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_sigmoid_invalid_output_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
