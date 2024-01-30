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

#include <iostream>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& op_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::add_outf(context, self, other, alpha, out);
}

Tensor& op_add_scalar_out(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::add_outf(context, self, other, alpha, out);
}

template <ScalarType DTYPE_A, ScalarType DTYPE_B, ScalarType DTYPE_OUT>
void test_add() {
  TensorFactory<DTYPE_A> tf_a;
  TensorFactory<DTYPE_B> tf_b;
  TensorFactory<DTYPE_OUT> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the sum.
  Tensor out = tf_out.zeros(sizes);

  // Add two tensors.
  op_add_out(
      tf_a.make(sizes, /*data=*/{1, 2, 4, 8}),
      tf_b.ones(sizes),
      /*alpha=*/1,
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, /*data=*/{2, 3, 5, 9}));
}

template <ScalarType DTYPE_A, ScalarType DTYPE_B>
void test_add_enumerate_out_types() {
  test_add<DTYPE_A, DTYPE_B, ScalarType::Half>();
  test_add<DTYPE_A, DTYPE_B, ScalarType::Float>();
  test_add<DTYPE_A, DTYPE_B, ScalarType::Double>();
  // Integral out type is only allowed if both inputs are integral types
  if (isIntegralType(DTYPE_A, false) && isIntegralType(DTYPE_B, false)) {
    test_add<DTYPE_A, DTYPE_B, ScalarType::Int>();
    test_add<DTYPE_A, DTYPE_B, ScalarType::Long>();
  }
}

template <ScalarType DTYPE_A>
void test_add_enumerate_b_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_add_enumerate_out_types<DTYPE_A, ScalarType::dtype>();

  ET_FORALL_REAL_TYPES_AND(Half, ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

void test_add_enumerate_a_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_add_enumerate_b_types<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES_AND(Half, ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

/**
 * Uses the function templates above to test all valid combinations of inputs
 * and output dtypes
 */
TEST(OpAddOutKernelTest, AllRealDtypesSupported) {
  test_add_enumerate_a_types();
}

// Common testing for adding two floating point Tensors.
template <ScalarType DTYPE>
void test_floating_point_add_out() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the sum.
  Tensor out = tf.zeros(sizes);

  // Add two tensors.
  op_add_out(
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}),
      tf.ones(sizes),
      /*alpha=*/1.1,
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{2.2, 3.3, 5.5, 9.9}));
}

TEST(OpAddOutKernelTest, FloatTensors) {
  test_floating_point_add_out<ScalarType::Float>();
}

TEST(OpAddOutKernelTest, DoubleTensors) {
  test_floating_point_add_out<ScalarType::Double>();
}

TEST(OpAddOutKernelTest, BoolAndIntInputTensor) {
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Int> tfi;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});
  Tensor b = tfi.make(sizes, /*data=*/{2, 4, 3, 3});

  Tensor out = tfi.zeros(sizes);

  op_add_out(a, b, /*alpha=*/1, out);
  EXPECT_TENSOR_EQ(out, tfi.make(sizes, {2, 5, 3, 4}));
}

TEST(OpAddOutKernelTest, BoolAndBoolInputTensor) {
  et_pal_init();
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});
  Tensor b = tf.make(sizes, /*data=*/{false, true, true, true});

  Tensor out = tf.zeros(sizes);

  op_add_out(a, b, /*alpha=*/1, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {false, true, true, true}));
}

TEST(OpAddOutKernelTest, BroadcastDimSizeIsOneAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.5721208453178406,
       0.9629082083702087,
       0.19517338275909424,
       0.4107270836830139,
       0.945562481880188,
       0.8788509368896484});
  Tensor y = tf.make({1, 2}, {0.7453382015228271, 0.3131374716758728});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.3174591064453125,
       1.2760456800460815,
       0.9405115842819214,
       0.7238645553588867,
       1.6909006834030151,
       1.191988468170166});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddOutKernelTest, BroadcastDimSizeMissingAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.5721208453178406,
       0.9629082083702087,
       0.19517338275909424,
       0.4107270836830139,
       0.945562481880188,
       0.8788509368896484});
  Tensor y = tf.make({2}, {0.7453382015228271, 0.3131374716758728});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.3174591064453125,
       1.2760456800460815,
       0.9405115842819214,
       0.7238645553588867,
       1.6909006834030151,
       1.191988468170166});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddOutKernelTest, BroadcastDimSizeIsOneBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.7453382015228271, 0.3131374716758728});
  Tensor y = tf.make(
      {3, 2},
      {0.5721208453178406,
       0.9629082083702087,
       0.19517338275909424,
       0.4107270836830139,
       0.945562481880188,
       0.8788509368896484});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.3174591064453125,
       1.2760456800460815,
       0.9405115842819214,
       0.7238645553588867,
       1.6909006834030151,
       1.191988468170166});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddOutKernelTest, BroadcastDimSizeMissingBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.7453382015228271, 0.3131374716758728});
  Tensor y = tf.make(
      {3, 2},
      {0.5721208453178406,
       0.9629082083702087,
       0.19517338275909424,
       0.4107270836830139,
       0.945562481880188,
       0.8788509368896484});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.3174591064453125,
       1.2760456800460815,
       0.9405115842819214,
       0.7238645553588867,
       1.6909006834030151,
       1.191988468170166});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddOutKernelTest, BroadcastSupported) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.zeros({5, 1, 3, 1});
  Tensor b = tf.ones({2, 1, 4});

  // Destination for the broadcasting sum. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({5, 2, 3, 4});

  Tensor ret = op_add_out(a, b, 1, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({5, 2, 3, 4}));
}

//
// Death Tests
//

TEST(OpAddOutKernelTest, IntInputsFloatAlphaDies) {
  // op_add_out() doesn't handle floating alpha for intergal inputs
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the op.
  Tensor out = tf.zeros(sizes);

  // Elementwise add operation on two integral tensor with floating alpha
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      op_add_out(tf.ones(sizes), tf.ones(sizes), /*alpha=*/.7, out));
}

TEST(OpAddOutKernelTest, BoolInputsFloatAlphaDies) {
  // op_add_out() doesn't handle floating alpha for intergal inputs
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the op.
  Tensor out = tf.zeros(sizes);

  // Elementwise add operation on two integral tensor with floating alpha
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      op_add_out(tf.ones(sizes), tf.ones(sizes), /*alpha=*/.7, out));
}

TEST(OpAddOutKernelTest, IntOutputWithFloatInputDies) {
  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Float> tff;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends.
  Tensor a = tfi.make(sizes, /*data=*/{2, 4, 3, 3});
  Tensor b = tff.make(sizes, /*data=*/{2, 4, 3, 3});

  // Destination for the sum.
  Tensor out = tfi.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_add_out(a, b, /*alpha=*/1, out));
}

TEST(OpAddOutKernelTest, BoolOutputWithIntegralInput) {
  // op_add_out() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Int> tfi;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends.
  Tensor a = tfi.make(sizes, /*data=*/{false, true, true, false});
  Tensor b = tfi.make(sizes, /*data=*/{2, 3, 4, 3});

  // Destination for the sum.
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_add_out(a, b, /*alpha=*/1, out));
}

TEST(OpAddOutKernelTest, MismatchedInputShapesDies) {
  TensorFactory<ScalarType::Int> tf;

  // Addends with different shapes.
  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor b = tf.ones(/*sizes=*/{2, 2});

  // Destination for the sum; matches the shape of one of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Adding the two mismatched tensors should cause an assertion and kill the
  // test process.
  ET_EXPECT_KERNEL_FAILURE(op_add_out(a, b, /*unused=*/0, out));
}

TEST(OpAddOutKernelTest, MismatchedOutputShapesDies) {
  if (SupportedFeatures::get()->output_resize) {
    GTEST_SKIP()
        << "The current kernel supports implicitly resizing output tensor";
  }

  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends with the same shapes.
  Tensor a = tf.ones(sizes);
  Tensor b = tf.ones(sizes);

  // Destination with a different shape.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Adding the tensors into a mismatched output should cause an assertion and
  // kill the test process.
  ET_EXPECT_KERNEL_FAILURE(op_add_out(a, b, /*unused=*/0, out));
}

TEST(OpAddOutKernelTest, SimpleGeneratedCase) {
  et_pal_init();

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
  Tensor y = tf.make(
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
      {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
       2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04024535417556763,
       0.6475827097892761,
       0.9623860716819763,
       0.6206040978431702,
       0.47623592615127563,
       0.4509747624397278});
  Tensor y = tf.make(
      {3, 2},
      {0.7232733964920044,
       0.3614498972892761,
       0.15757757425308228,
       0.9975225925445557,
       0.09227871894836426,
       0.3320664167404175});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.763518750667572,
       1.0090326070785522,
       1.1199636459350586,
       1.618126630783081,
       0.5685146450996399,
       0.7830411791801453});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04024535417556763,
       0.6475827097892761,
       0.9623860716819763,
       0.6206040978431702,
       0.47623592615127563,
       0.4509747624397278});
  Tensor y = tf.make(
      {3, 2},
      {0.7232733964920044,
       0.3614498972892761,
       0.15757757425308228,
       0.9975225925445557,
       0.09227871894836426,
       0.3320664167404175});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.763518750667572,
       1.0090326070785522,
       1.1199636459350586,
       1.618126630783081,
       0.5685146450996399,
       0.7830411791801453});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddOutKernelTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04024535417556763,
       0.6475827097892761,
       0.9623860716819763,
       0.6206040978431702,
       0.47623592615127563,
       0.4509747624397278});
  Tensor y = tf.make(
      {3, 2},
      {0.7232733964920044,
       0.3614498972892761,
       0.15757757425308228,
       0.9975225925445557,
       0.09227871894836426,
       0.3320664167404175});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.763518750667572,
       1.0090326070785522,
       1.1199636459350586,
       1.618126630783081,
       0.5685146450996399,
       0.7830411791801453});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_add_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpAddScalarOutKernelTest, SanityCheck) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_add_scalar_out(tf.make(sizes, {1, 2, 4, 8}), true, /*alpha=*/2, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {3, 4, 6, 10}));
}

TEST(OpAddScalarOutKernelTest, OptimizedSanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_add_scalar_out(
      tf.make(sizes, {1.3, 2.1, 4.6, 8.2}), 1.9, /*alpha=*/2.8, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, {6.62, 7.42, 9.92, 13.52}));
}
