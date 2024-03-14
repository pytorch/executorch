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
#include <algorithm>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& op_mul_out(const Tensor& self, const Tensor& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::mul_outf(context, self, other, out);
}

Tensor&
op_mul_scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::mul_outf(context, self, other, out);
}

//
// Correctness Tests
//

// Common testing for multipling two integer Tensors
template <ScalarType DTYPE_A, ScalarType DTYPE_B, ScalarType DTYPE_OUT>
void test_mul() {
  TensorFactory<DTYPE_A> tf_a;
  TensorFactory<DTYPE_B> tf_b;
  TensorFactory<DTYPE_OUT> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the mul.
  Tensor out = tf_out.zeros(sizes);

  // Multiply two tensors
  op_mul_out(tf_a.make(sizes, /*data=*/{1, 2, 4, 8}), tf_b.ones(sizes), out);
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, /*data=*/{1, 2, 4, 8}));

  op_mul_out(tf_a.make(sizes, /*data=*/{1, 2, 4, 8}), tf_b.zeros(sizes), out);
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, /*data=*/{0, 0, 0, 0}));

  op_mul_out(
      tf_a.make(sizes, /*data=*/{1, 2, 4, 8}),
      tf_b.make(sizes, /*data=*/{1, 2, 4, 8}),
      out);
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, /*data=*/{1, 4, 16, 64}));
}

template <ScalarType DTYPE_A, ScalarType DTYPE_B>
void test_mul_enumerate_out_types() {
  test_mul<DTYPE_A, DTYPE_B, ScalarType::Half>();
  test_mul<DTYPE_A, DTYPE_B, ScalarType::Float>();
  test_mul<DTYPE_A, DTYPE_B, ScalarType::Double>();
  // Integral out type is only allowed if both inputs are integral types
  if (isIntegralType(DTYPE_A, false) && isIntegralType(DTYPE_B, false)) {
    test_mul<DTYPE_A, DTYPE_B, ScalarType::Int>();
    test_mul<DTYPE_A, DTYPE_B, ScalarType::Long>();
  }
}

template <ScalarType DTYPE_A>
void test_mul_enumerate_b_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_mul_enumerate_out_types<DTYPE_A, ScalarType::dtype>();

  ET_FORALL_REAL_TYPES_AND(Half, ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

void test_mul_enumerate_a_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_mul_enumerate_b_types<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES_AND(Half, ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

/**
 * Uses the function templates above to test all valid combinations of inputs
 * and output dtypes
 */
TEST(OpMulOutKernelTest, AllRealDtypesSupported) {
  test_mul_enumerate_a_types();
}

// Common testing for multipling two floating point Tensors
template <ScalarType DTYPE>
void test_floating_point_mul_out() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the mul.
  Tensor out = tf.zeros(sizes);

  // Multiply two tensors
  op_mul_out(
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}), tf.ones(sizes), out);
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}));

  op_mul_out(
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}), tf.zeros(sizes), out);
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{0.0, 0.0, 0.0, 0.0}));

  op_mul_out(
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}),
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}),
      out);
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{1.21, 4.84, 19.36, 77.44}));
}

TEST(OpMulOutKernelTest, FloatTensors) {
  test_floating_point_mul_out<ScalarType::Float>();
}

TEST(OpMulOutKernelTest, DoubleTensors) {
  test_floating_point_mul_out<ScalarType::Double>();
}

TEST(OpMulOutKernelTest, BoolTensors) {
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the mul.
  Tensor out = tf.zeros(sizes);

  // Multiply two tensors
  op_mul_out(
      tf.make(sizes, /*data=*/{true, false, true, true}), tf.ones(sizes), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{true, false, true, true}));

  op_mul_out(
      tf.make(sizes, /*data=*/{true, false, true, true}), tf.zeros(sizes), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{false, false, false, false}));

  op_mul_out(
      tf.make(sizes, /*data=*/{true, false, true, true}),
      tf.make(sizes, /*data=*/{false, false, true, false}),
      out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{false, false, true, false}));
}

// Mismatched shape tests.
TEST(OpMulOutKernelTest, MismatchedInputShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen currently supports mismatched shapes";
  }

  TensorFactory<ScalarType::Int> tf;

  // Input tensors with different shapes.
  Tensor a = tf.ones(/*sizes=*/{1, 2});
  Tensor b = tf.ones(/*sizes=*/{2, 2});

  // Output tensor; matches the shape of one of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Multiplying the two mismatched tensors should cause an assertion and kill
  // the test process.
  ET_EXPECT_KERNEL_FAILURE(op_mul_out(a, b, out));
}

// Broadcast tensor b's size to tensor a's size
TEST(OpMulOutKernelTest, BroadcastA2BTest) {
  TensorFactory<ScalarType::Int> tf_a;

  // a and b of different shapes
  Tensor a = tf_a.make({2, 2}, /*data=*/{1, 2, 3, 4});
  Tensor b = tf_a.make({2}, /*data=*/{2, 2});

  // Destination for output of mul.
  Tensor out = tf_a.zeros({2, 2});

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      op_mul_out(a, b, out), tf_a.make({2, 2}, /*data=*/{2, 4, 6, 8}));
}

// Broadcast tensor a's size to tensor b's size
TEST(OpMulOutKernelTest, BroadcastB2ATest) {
  TensorFactory<ScalarType::Int> tf_a;

  // a and b of different shapes
  Tensor a = tf_a.make({2}, /*data=*/{2, 2});
  Tensor b = tf_a.make({2, 2}, /*data=*/{1, 2, 3, 4});

  // Destination for output of mul.
  Tensor out = tf_a.zeros({2, 2});

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      op_mul_out(a, b, out), tf_a.make({2, 2}, /*data=*/{2, 4, 6, 8}));
}

// Broadcast tensor a and b's size to a new size c.
TEST(OpMulOutKernelTest, BroadcastAB2CTest) {
  TensorFactory<ScalarType::Int> tf_a;

  // a and b of different shapes
  Tensor a = tf_a.make({2, 1}, /*data=*/{1, 2});
  Tensor b = tf_a.make({2, 1, 2}, /*data=*/{1, 2, 3, 4});

  // Destination for output of mul.
  Tensor out = tf_a.zeros({2, 2, 2});

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      op_mul_out(a, b, out),
      tf_a.make({2, 2, 2}, /*data=*/{1, 2, 2, 4, 3, 4, 6, 8}));
}

TEST(OpMaskedFillTest, ScalarInputBroadcastTest) {
  TensorFactory<ScalarType::Int> tf_a;

  // a is a 1d tensor and b is a scalar
  Tensor a = tf_a.make({2}, /*data=*/{2, 2});
  Tensor b = tf_a.make({}, /*data=*/{2});

  // Destination for output of mul.
  Tensor out = tf_a.make({2}, /*data=*/{2, 2});
  Tensor expected = tf_a.make({2}, /*data=*/{4, 4});

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
}

TEST(OpMulOutKernelTest, MismatchedOutputShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen currently supports mismatched shapes";
  }

  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Input tensors with the same shapes.
  Tensor a = tf.ones(sizes);
  Tensor b = tf.ones(sizes);

  // Output tensor with a different shape.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Multiplying the tensors into a mismatched output should cause an assertion
  // and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(op_mul_out(a, b, out));
}

TEST(OpMulOutKernelTest, BroadcastDimSizeIsOneAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.3200607895851135,
       0.029979348182678223,
       0.27112698554992676,
       0.15423381328582764,
       0.6920414566993713,
       0.005174398422241211});
  Tensor y = tf.make({1, 2}, {0.9711773991584778, 0.8632034063339233});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.3108358085155487,
       0.02587827481329441,
       0.2633123993873596,
       0.13313515484333038,
       0.672095000743866,
       0.004466558340936899});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_mul_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpMulOutKernelTest, BroadcastDimSizeMissingAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.3200607895851135,
       0.029979348182678223,
       0.27112698554992676,
       0.15423381328582764,
       0.6920414566993713,
       0.005174398422241211});
  Tensor y = tf.make({2}, {0.9711773991584778, 0.8632034063339233});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.3108358085155487,
       0.02587827481329441,
       0.2633123993873596,
       0.13313515484333038,
       0.672095000743866,
       0.004466558340936899});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_mul_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpMulOutKernelTest, BroadcastDimSizeIsOneBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.9711773991584778, 0.8632034063339233});
  Tensor y = tf.make(
      {3, 2},
      {0.3200607895851135,
       0.029979348182678223,
       0.27112698554992676,
       0.15423381328582764,
       0.6920414566993713,
       0.005174398422241211});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.3108358085155487,
       0.02587827481329441,
       0.2633123993873596,
       0.13313515484333038,
       0.672095000743866,
       0.004466558340936899});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_mul_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpMulOutKernelTest, BroadcastDimSizeMissingBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.9711773991584778, 0.8632034063339233});
  Tensor y = tf.make(
      {3, 2},
      {0.3200607895851135,
       0.029979348182678223,
       0.27112698554992676,
       0.15423381328582764,
       0.6920414566993713,
       0.005174398422241211});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.3108358085155487,
       0.02587827481329441,
       0.2633123993873596,
       0.13313515484333038,
       0.672095000743866,
       0.004466558340936899});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_mul_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpMulOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6910695433616638,
       0.6540696620941162,
       0.8072559237480164,
       0.8218746185302734,
       0.9193597435951233,
       0.4525110721588135});
  Tensor y = tf.make(
      {3, 2},
      {0.9212601184844971,
       0.2030404806137085,
       0.34644562005996704,
       0.4489826560020447,
       0.5666958689689636,
       0.5006863474845886});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.636654794216156,
       0.13280262053012848,
       0.27967026829719543,
       0.3690074384212494,
       0.5209973454475403,
       0.2265661209821701});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_mul_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpMulOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6910695433616638,
       0.6540696620941162,
       0.8072559237480164,
       0.8218746185302734,
       0.9193597435951233,
       0.4525110721588135});
  Tensor y = tf.make(
      {3, 2},
      {0.9212601184844971,
       0.2030404806137085,
       0.34644562005996704,
       0.4489826560020447,
       0.5666958689689636,
       0.5006863474845886});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.636654794216156,
       0.13280262053012848,
       0.27967026829719543,
       0.3690074384212494,
       0.5209973454475403,
       0.2265661209821701});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_mul_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpMulOutKernelTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6910695433616638,
       0.6540696620941162,
       0.8072559237480164,
       0.8218746185302734,
       0.9193597435951233,
       0.4525110721588135});
  Tensor y = tf.make(
      {3, 2},
      {0.9212601184844971,
       0.2030404806137085,
       0.34644562005996704,
       0.4489826560020447,
       0.5666958689689636,
       0.5006863474845886});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.636654794216156,
       0.13280262053012848,
       0.27967026829719543,
       0.3690074384212494,
       0.5209973454475403,
       0.2265661209821701});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_mul_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST(OpMulScalarOutKernelTest, SanityCheck) {
  TensorFactory<ScalarType::Bool> tf_a;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf_out.zeros(sizes);

  op_mul_scalar_out(tf_a.make(sizes, {true, false, true, false}), 2.3, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {2.3, 0.0, 2.3, 0.0}));
}

TEST(OpMulScalarOutKernelTest, OptimizedSanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_mul_scalar_out(tf.make(sizes, {1.3, 2.1, 4.6, 8.2}), 2.0, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, {2.6, 4.2, 9.2, 16.4}));
}
