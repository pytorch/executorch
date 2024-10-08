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
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;
using torch::executor::testing::SupportedFeatures;
namespace etrt = executorch::runtime;

class OpMulOutTest : public OperatorTest {
 protected:
  Tensor& op_mul_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::aten::mul_outf(context_, self, other, out);
  }

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
    if (etrt::isIntegralType(DTYPE_A, false) &&
        etrt::isIntegralType(DTYPE_B, false)) {
      test_mul<DTYPE_A, DTYPE_B, ScalarType::Int>();
      test_mul<DTYPE_A, DTYPE_B, ScalarType::Long>();
    }
  }

  template <ScalarType DTYPE_A>
  void test_mul_enumerate_b_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_mul_enumerate_out_types<DTYPE_A, ScalarType::dtype>();

    ET_FORALL_REALHBF16_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
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
        tf.make(sizes, /*data=*/{1.25, 2.5, 4.75, 8.875}), tf.ones(sizes), out);
    EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{1.25, 2.5, 4.75, 8.875}));

    op_mul_out(
        tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}), tf.zeros(sizes), out);
    EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{0.0, 0.0, 0.0, 0.0}));

    op_mul_out(
        tf.make(sizes, /*data=*/{1.25, 2.5, 4.75, 8.875}),
        tf.make(sizes, /*data=*/{1.25, 2.5, 4.75, 8.875}),
        out);
    EXPECT_TENSOR_CLOSE(
        out, tf.make(sizes, /*data=*/{1.5625, 6.25, 22.5625, 78.765625}));
  }

  void test_mul_enumerate_a_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_mul_enumerate_b_types<ScalarType::dtype>();

    ET_FORALL_REALHBF16_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  template <ScalarType DTYPE>
  void test_optimized_path_ignores_leading_1_dimensions() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes1 = {1, 1, 2, 2};
    const std::vector<int32_t> sizes2 = {1, 2, 2};

    // Destination for the mul.
    Tensor out = tf.zeros(sizes1);

    // Multiply two tensors
    op_mul_out(
        tf.make(sizes1, /*data=*/{1.1, 2.2, 4.4, 8.8}), tf.ones(sizes2), out);
    EXPECT_TENSOR_CLOSE(out, tf.make(sizes1, /*data=*/{1.1, 2.2, 4.4, 8.8}));
  }

  template <ScalarType DTYPE>
  void test_broadcast_a2b() {
    TensorFactory<DTYPE> tf_a;

    std::vector<std::vector<int32_t>> b_sizeses = {
        {2},
        {1, 2},
    };
    for (const auto& b_sizes : b_sizeses) {
      // a and b of different shapes
      Tensor a = tf_a.make({2, 2}, /*data=*/{1, 2, 3, 4});
      Tensor b = tf_a.make(b_sizes, /*data=*/{2, 2});

      // Destination for output of mul.
      Tensor out = tf_a.zeros({2, 2});

      // Check that it matches the expected output.
      EXPECT_TENSOR_CLOSE(
          op_mul_out(a, b, out), tf_a.make({2, 2}, /*data=*/{2, 4, 6, 8}));
    }
  }

  template <ScalarType DTYPE>
  void test_broadcast_b2a() {
    TensorFactory<DTYPE> tf_a;
    // a and b of different shapes
    Tensor a = tf_a.make({2}, /*data=*/{2, 2});
    Tensor b = tf_a.make({2, 2}, /*data=*/{1, 2, 3, 4});

    // Destination for output of mul.
    Tensor out = tf_a.zeros({2, 2});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(
        op_mul_out(a, b, out), tf_a.make({2, 2}, /*data=*/{2, 4, 6, 8}));
  }

  template <ScalarType DTYPE>
  void test_scalar_input_broadcast() {
    TensorFactory<DTYPE> tf_a;

    // a is a 1d tensor and b is a scalar
    Tensor a = tf_a.make({2}, /*data=*/{2, 2});
    Tensor b = tf_a.make({}, /*data=*/{2});

    // Destination for output of mul.
    Tensor out = tf_a.make({2}, /*data=*/{2, 2});
    Tensor expected = tf_a.make({2}, /*data=*/{4, 4});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);
  }

  template <ScalarType DTYPE>
  void test_both_scalar_input_broadcast() {
    TensorFactory<DTYPE> tf_a;

    // a is a rank-1 scalar and b is a rank-0 scalar
    Tensor a = tf_a.make({1}, /*data=*/{2});
    Tensor b = tf_a.make({}, /*data=*/{2});

    // Destination for output of mul.
    Tensor out = tf_a.make({1}, /*data=*/{2});
    Tensor expected = tf_a.make({1}, /*data=*/{4});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);
  }
};

class OpMulScalarOutTest : public OperatorTest {
 protected:
  Tensor&
  op_mul_scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    return torch::executor::aten::mul_outf(context_, self, other, out);
  }
};

//
// Correctness Tests
//

/**
 * Uses the function templates above to test all valid combinations of
 * inputs*and output dtypes*/
TEST_F(OpMulOutTest, AllRealDtypesSupported) {
  test_mul_enumerate_a_types();
}

TEST_F(OpMulOutTest, FloatTensors) {
  test_floating_point_mul_out<ScalarType::Float>();
}

TEST_F(OpMulOutTest, DoubleTensors) {
  test_floating_point_mul_out<ScalarType::Double>();
}

TEST_F(OpMulOutTest, HalfTensors) {
  test_floating_point_mul_out<ScalarType::Half>();
}

TEST_F(OpMulOutTest, BFloat16Tensors) {
  test_floating_point_mul_out<ScalarType::BFloat16>();
}

TEST_F(OpMulOutTest, BoolTensors) {
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

TEST_F(OpMulOutTest, OptimizedPathIgnoresLeading1Dimensions) {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_optimized_path_ignores_leading_1_dimensions<ScalarType::dtype>();

  ET_FORALL_FLOATHBF16_TYPES(ENUMERATE_TEST_ENTRY);

#undef ENUMERATE_TEST_ENTRY
}

// Mismatched shape tests.
TEST_F(OpMulOutTest, MismatchedNonBroadcastableInputShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen currently supports mismatched shapes";
  }

  TensorFactory<ScalarType::Int> tf;

  // Input tensors with different shapes.
  Tensor a = tf.ones(/*sizes=*/{4, 2});
  Tensor b = tf.ones(/*sizes=*/{2, 2});

  // Output tensor; matches the shape of one of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{8});

  // Multiplying the two mismatched tensors should cause an assertion and kill
  // the test process.
  ET_EXPECT_KERNEL_FAILURE(context_, op_mul_out(a, b, out));
}

// Broadcast tensor b's size to tensor a's size
TEST_F(OpMulOutTest, BroadcastA2BTest) {
  test_broadcast_a2b<ScalarType::Int>();
  test_broadcast_a2b<ScalarType::Half>();
  test_broadcast_a2b<ScalarType::BFloat16>();
}

// Broadcast tensor a's size to tensor b's size
TEST_F(OpMulOutTest, BroadcastB2ATest) {
  test_broadcast_b2a<ScalarType::Int>();
  test_broadcast_b2a<ScalarType::Half>();
  test_broadcast_b2a<ScalarType::BFloat16>();
}

// Broadcast tensor a and b's size to a new size c.
TEST_F(OpMulOutTest, BroadcastAB2CTest) {
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

TEST_F(OpMulOutTest, ScalarInputBroadcastTest) {
  test_scalar_input_broadcast<ScalarType::Int>();
  test_scalar_input_broadcast<ScalarType::Half>();
  test_scalar_input_broadcast<ScalarType::BFloat16>();
}

TEST_F(OpMulOutTest, BothScalarInputBroadcastTest) {
  test_both_scalar_input_broadcast<ScalarType::Int>();
  test_both_scalar_input_broadcast<ScalarType::Half>();
  test_both_scalar_input_broadcast<ScalarType::BFloat16>();
}

TEST_F(OpMulOutTest, MismatchedOutputShapesDies) {
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
  ET_EXPECT_KERNEL_FAILURE(context_, op_mul_out(a, b, out));
}

TEST_F(OpMulOutTest, BroadcastDimSizeIsOneAB) {
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

TEST_F(OpMulOutTest, BroadcastDimSizeMissingAB) {
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

TEST_F(OpMulOutTest, BroadcastDimSizeIsOneBA) {
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

TEST_F(OpMulOutTest, BroadcastDimSizeMissingBA) {
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

TEST_F(OpMulOutTest, DynamicShapeUpperBoundSameAsExpected) {
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

TEST_F(OpMulOutTest, DynamicShapeUpperBoundLargerThanExpected) {
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

TEST_F(OpMulOutTest, DynamicShapeUnbound) {
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

TEST_F(OpMulScalarOutTest, SanityCheck) {
  TensorFactory<ScalarType::Bool> tf_a;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf_out.zeros(sizes);

  op_mul_scalar_out(tf_a.make(sizes, {true, false, true, false}), 2.3, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {2.3, 0.0, 2.3, 0.0}));
}

TEST_F(OpMulScalarOutTest, OptimizedSanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_mul_scalar_out(tf.make(sizes, {1.3, 2.1, 4.6, 8.2}), 2.0, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, {2.6, 4.2, 9.2, 16.4}));
}

TEST_F(OpMulScalarOutTest, HalfSanityCheck) {
  TensorFactory<ScalarType::Half> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_mul_scalar_out(tf.make(sizes, {1.3, 2.1, 4.6, 8.2}), 2.0, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, {2.6, 4.2, 9.2, 16.4}));
}

TEST_F(OpMulScalarOutTest, BFloat16SanityCheck) {
  TensorFactory<ScalarType::BFloat16> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_mul_scalar_out(tf.make(sizes, {1.3, 2.1, 4.6, 8.2}), 2.0, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, {2.6, 4.2, 9.2, 16.4}));
}
