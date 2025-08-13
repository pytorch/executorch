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

  // Common testing for multiplying two integer Tensors
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

    out = tf_out.zeros({18});
    op_mul_out(tf_a.full({18}, 4), tf_b.full({18}, 2), out);
    EXPECT_TENSOR_EQ(out, tf_out.full({18}, 8));
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
  void test_broadcast_3D() {
    TensorFactory<DTYPE> tf_a;

    Tensor a =
        tf_a.make({2, 2, 3}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor b = tf_a.make({2, 1, 3}, /*data=*/{2, 3, 4, 5, 6, 7});

    // Destination for output of mul.
    Tensor out =
        tf_a.make({2, 2, 3}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor expected = tf_a.make(
        {2, 2, 3}, /*data=*/{2, 6, 12, 8, 15, 24, 35, 48, 63, 50, 66, 84});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);
  }

  template <ScalarType DTYPE>
  void test_broadcast_4D() {
    TensorFactory<DTYPE> tf_a;

    Tensor a = tf_a.make(
        {2, 2, 3, 5},
        /*data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                  46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60});
    Tensor b = tf_a.make(
        {2, 1, 3, 5},
        /*data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

    // Destination for output of mul.
    Tensor out = tf_a.zeros({2, 2, 3, 5});
    Tensor expected = tf_a.make(
        {2, 2, 3, 5},
        /*data=*/{1,    4,    9,    16,   25,   36,   49,   64,   81,   100,
                  121,  144,  169,  196,  225,  16,   34,   54,   76,   100,
                  126,  154,  184,  216,  250,  286,  324,  364,  406,  450,
                  496,  544,  594,  646,  700,  756,  814,  874,  936,  1000,
                  1066, 1134, 1204, 1276, 1350, 736,  799,  864,  931,  1000,
                  1071, 1144, 1219, 1296, 1375, 1456, 1539, 1624, 1711, 1800});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);

    b = tf_a.make(
        {2, 2, 1, 5}, /*data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    out = tf_a.zeros({2, 2, 3, 5});
    expected = tf_a.make(
        {2, 2, 3, 5},
        /*data=*/{1,   4,   9,   16,   25,   6,   14,  24,   36,   50,
                  11,  24,  39,  56,   75,   96,  119, 144,  171,  200,
                  126, 154, 184, 216,  250,  156, 189, 224,  261,  300,
                  341, 384, 429, 476,  525,  396, 444, 494,  546,  600,
                  451, 504, 559, 616,  675,  736, 799, 864,  931,  1000,
                  816, 884, 954, 1026, 1100, 896, 969, 1044, 1121, 1200});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);
  }

  template <ScalarType DTYPE>
  void test_broadcast_last_dim() {
    TensorFactory<DTYPE> tf_a;

    Tensor a =
        tf_a.make({4, 3}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor b = tf_a.make({4, 1}, /*data=*/{2, 3, 4, 5});

    // Destination for output of mul.
    Tensor out = tf_a.zeros({4, 3});
    Tensor expected = tf_a.make(
        {4, 3}, /*data=*/{2, 4, 6, 12, 15, 18, 28, 32, 36, 50, 55, 60});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);

    a = tf_a.make({2, 2, 3}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    b = tf_a.make({2, 2, 1}, /*data=*/{2, 3, 4, 5});

    // Destination for output of mul.
    out = tf_a.zeros({2, 2, 3});
    expected = tf_a.make(
        {2, 2, 3}, /*data=*/{2, 4, 6, 12, 15, 18, 28, 32, 36, 50, 55, 60});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);

    a = tf_a.make(
        {2, 2, 3, 5},
        /*data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                  46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60});
    b = tf_a.make(
        {2, 2, 3, 1},
        /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // Destination for output of mul.
    out = tf_a.zeros({2, 2, 3, 5});
    expected = tf_a.make(
        {2, 2, 3, 5},
        /*data=*/{1,   2,   3,   4,   5,   12,  14,  16,  18,  20,  33,  36,
                  39,  42,  45,  64,  68,  72,  76,  80,  105, 110, 115, 120,
                  125, 156, 162, 168, 174, 180, 217, 224, 231, 238, 245, 288,
                  296, 304, 312, 320, 369, 378, 387, 396, 405, 460, 470, 480,
                  490, 500, 561, 572, 583, 594, 605, 672, 684, 696, 708, 720});

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(op_mul_out(a, b, out), expected);
    EXPECT_TENSOR_CLOSE(op_mul_out(b, a, out), expected);
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

  template <typename CTYPE, ScalarType DTYPE>
  void test_complex_dtype() {
    TensorFactory<DTYPE> tf;
    const std::vector<int32_t> sizes = {2, 2};

    // Create complex tensors with real and imaginary parts
    Tensor x =
        tf.make(sizes, {CTYPE(1, 2), CTYPE(3, 4), CTYPE(5, 6), CTYPE(7, 8)});

    Tensor y =
        tf.make(sizes, {CTYPE(2, 3), CTYPE(4, 5), CTYPE(6, 7), CTYPE(8, 9)});

    // Expected result: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
    // (1+2i) * (2+3i) = (1*2-2*3) + (1*3+2*2)i = -4 + 7i
    // (3+4i) * (4+5i) = (3*4-4*5) + (3*5+4*4)i = -8 + 31i
    // (5+6i) * (6+7i) = (5*6-6*7) + (5*7+6*6)i = -12 + 71i
    // (7+8i) * (8+9i) = (7*8-8*9) + (7*9+8*8)i = -16 + 127i
    Tensor expected = tf.make(
        sizes, {CTYPE(-4, 7), CTYPE(-8, 31), CTYPE(-12, 71), CTYPE(-16, 127)});

    Tensor out = tf.make(
        {2, 2},
        {
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
        });
    op_mul_out(x, y, out);
    EXPECT_TENSOR_CLOSE(out, expected);
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

TEST_F(OpMulOutTest, BroadcastNDTest) {
  // Test 3D tensors
  test_broadcast_3D<ScalarType::Float>();
  test_broadcast_3D<ScalarType::Half>();
  test_broadcast_3D<ScalarType::BFloat16>();

  // Test 4D tensors
  test_broadcast_4D<ScalarType::Float>();
  test_broadcast_4D<ScalarType::Half>();
  test_broadcast_4D<ScalarType::BFloat16>();

  // Test broadcasting on the last dimension
  test_broadcast_last_dim<ScalarType::Float>();
  test_broadcast_last_dim<ScalarType::Half>();
  test_broadcast_last_dim<ScalarType::BFloat16>();
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

TEST_F(OpMulOutTest, AllComplexDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_complex_dtype<ctype, ScalarType::dtype>();
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    ET_FORALL_COMPLEX_TYPES(TEST_ENTRY);
  } else {
    ET_FORALL_COMPLEXH_TYPES(TEST_ENTRY);
  }
#undef TEST_ENTRY
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

// >>> torch.ops.aten.mul(torch.tensor([100], dtype=torch.int8),
// torch.tensor([100], dtype=torch.int8), out=torch.zeros([1],
// dtype=torch.long)) tensor([16])
TEST_F(OpMulOutTest, MixedIntegerDtypeMatchesATen) {
  TensorFactory<ScalarType::Char> tf_in;
  TensorFactory<ScalarType::Long> tf_out;

  Tensor in = tf_in.make({1}, {100});
  Tensor out = tf_out.zeros({1});
  Tensor ret = op_mul_out(in, in, out);

  Tensor expected = tf_out.make({1}, {16});
  EXPECT_TENSOR_CLOSE(out, expected);
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

// Tests for broadcast handling fix: when tensor dimensions don't match,
// the output should be resized to match the tensor with higher dimensionality
TEST_F(OpMulOutTest, BroadcastDimensionMismatchFix) {
  TensorFactory<ScalarType::Float> tf;

  // Test case: tensor a of size [6] and b of size [1, 1, 6]
  // Expected output should be [1, 1, 6], not [6]
  Tensor a = tf.make({6}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor b = tf.make({1, 1, 6}, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0});

  // Create output tensor with expected broadcast shape [1, 1, 6]
  Tensor out = tf.zeros({1, 1, 6});

  // Call the mul function
  Tensor& result = op_mul_out(a, b, out);

  // Verify the output shape is [1, 1, 6]
  EXPECT_EQ(result.dim(), 3);
  EXPECT_EQ(result.size(0), 1);
  EXPECT_EQ(result.size(1), 1);
  EXPECT_EQ(result.size(2), 6);

  // Verify the values are correct (element-wise multiplication with
  // broadcasting)
  Tensor expected = tf.make({1, 1, 6}, {2.0, 4.0, 6.0, 8.0, 10.0, 12.0});
  EXPECT_TENSOR_CLOSE(result, expected);
}

TEST_F(OpMulOutTest, BroadcastDimensionMismatchReversed) {
  TensorFactory<ScalarType::Float> tf;

  // Test case: tensor a of size [1, 1, 6] and b of size [6]
  // Expected output should be [1, 1, 6]
  Tensor a = tf.make({1, 1, 6}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor b = tf.make({6}, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0});

  // Create output tensor with expected broadcast shape [1, 1, 6]
  Tensor out = tf.zeros({1, 1, 6});

  // Call the mul function
  Tensor& result = op_mul_out(a, b, out);

  // Verify the output shape is [1, 1, 6]
  EXPECT_EQ(result.dim(), 3);
  EXPECT_EQ(result.size(0), 1);
  EXPECT_EQ(result.size(1), 1);
  EXPECT_EQ(result.size(2), 6);

  // Verify the values are correct (element-wise multiplication with
  // broadcasting)
  Tensor expected = tf.make({1, 1, 6}, {2.0, 4.0, 6.0, 8.0, 10.0, 12.0});
  EXPECT_TENSOR_CLOSE(result, expected);
}

TEST_F(OpMulOutTest, BroadcastDimensionMismatchWithDifferentTypes) {
  // Test the same broadcast fix with different data types
  TensorFactory<ScalarType::Half> tf_half;
  TensorFactory<ScalarType::BFloat16> tf_bf16;
  TensorFactory<ScalarType::Int> tf_int;

  // Test with Half precision
  {
    Tensor a = tf_half.make({4}, {1.0, 2.0, 3.0, 4.0});
    Tensor b = tf_half.make({1, 1, 4}, {2.0, 2.0, 2.0, 2.0});
    Tensor out = tf_half.zeros({1, 1, 4});

    Tensor& result = op_mul_out(a, b, out);
    EXPECT_EQ(result.dim(), 3);
    EXPECT_EQ(result.size(0), 1);
    EXPECT_EQ(result.size(1), 1);
    EXPECT_EQ(result.size(2), 4);

    Tensor expected = tf_half.make({1, 1, 4}, {2.0, 4.0, 6.0, 8.0});
    EXPECT_TENSOR_CLOSE(result, expected);
  }

  // Test with BFloat16
  {
    Tensor a = tf_bf16.make({4}, {1.0, 2.0, 3.0, 4.0});
    Tensor b = tf_bf16.make({1, 1, 4}, {2.0, 2.0, 2.0, 2.0});
    Tensor out = tf_bf16.zeros({1, 1, 4});

    Tensor& result = op_mul_out(a, b, out);
    EXPECT_EQ(result.dim(), 3);
    EXPECT_EQ(result.size(0), 1);
    EXPECT_EQ(result.size(1), 1);
    EXPECT_EQ(result.size(2), 4);

    Tensor expected = tf_bf16.make({1, 1, 4}, {2.0, 4.0, 6.0, 8.0});
    EXPECT_TENSOR_CLOSE(result, expected);
  }

  // Test with Int
  {
    Tensor a = tf_int.make({4}, {1, 2, 3, 4});
    Tensor b = tf_int.make({1, 1, 4}, {2, 2, 2, 2});
    Tensor out = tf_int.zeros({1, 1, 4});

    Tensor& result = op_mul_out(a, b, out);
    EXPECT_EQ(result.dim(), 3);
    EXPECT_EQ(result.size(0), 1);
    EXPECT_EQ(result.size(1), 1);
    EXPECT_EQ(result.size(2), 4);

    Tensor expected = tf_int.make({1, 1, 4}, {2, 4, 6, 8});
    EXPECT_TENSOR_EQ(result, expected);
  }
}
