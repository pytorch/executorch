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
#include <cmath>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

namespace {

class OpDivOutTest : public OperatorTest {
 protected:
  Tensor& op_div_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return torch::executor::aten::div_outf(context_, a, b, out);
  }

  template <ScalarType DTYPE_A, ScalarType DTYPE_B, ScalarType DTYPE_OUT>
  void test_div() {
    TensorFactory<DTYPE_A> tf_a;
    TensorFactory<DTYPE_B> tf_b;
    TensorFactory<DTYPE_OUT> tf_out;

    const std::vector<int32_t> sizes = {2, 2};

    Tensor out = tf_out.zeros(sizes);

    // Valid input should give the expected output
    op_div_out(
        tf_a.make(sizes, /*data=*/{1, 2, 4, 8}),
        tf_b.make(sizes, /*data=*/{8, 4, 2, 1}),
        out);

    EXPECT_TENSOR_CLOSE(out, tf_out.make(sizes, /*data=*/{0.125, 0.5, 2, 8}));
  }

  template <ScalarType DTYPE_A, ScalarType DTYPE_B>
  void test_div_enumerate_out_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_div<DTYPE_A, DTYPE_B, ScalarType::dtype>();

    ET_FORALL_FLOAT_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  template <ScalarType DTYPE_A>
  void test_div_enumerate_b_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_div_enumerate_out_types<DTYPE_A, ScalarType::dtype>();

    ET_FORALL_REAL_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  template <ScalarType OUTPUT_DTYPE>
  void test_div_invalid_output_dtype_dies() {
    TensorFactory<ScalarType::Float> tf_float;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    Tensor a = tf_float.ones(sizes);
    Tensor b = tf_float.ones(sizes);
    Tensor out = tf_out.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_div_out(a, b, out));
  }

  /**
   * Common testing for div operator, for float output types
   */
  void test_div_enumerate_a_types();
};

template <>
void OpDivOutTest::
    test_div<ScalarType::Float, ScalarType::Float, ScalarType::Float>() {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 5};

  // Invalid divisor input zero should die
  Tensor out = tf.zeros(sizes);

  // Valid input should give the expected output
  op_div_out(
      tf.make(sizes, /*data=*/{1, 2, 4, 8, INFINITY, -INFINITY, NAN, 1, 1, 1}),
      tf.make(
          sizes,
          /*data=*/
          {8, 0, 2, 1, INFINITY, -INFINITY, NAN, INFINITY, -INFINITY, NAN}),
      out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          sizes, /*data=*/{0.125, INFINITY, 2, 8, NAN, NAN, NAN, 0, 0, NAN}));
}

template <>
void OpDivOutTest::
    test_div<ScalarType::Bool, ScalarType::Float, ScalarType::Float>() {
  TensorFactory<ScalarType::Bool> tf_b;
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Invalid divisor input zero should die
  Tensor out = tf.zeros(sizes);

  // Valid input should give the expected output
  op_div_out(
      tf_b.make(sizes, /*data=*/{1, 1, 1, 1}),
      tf.make(sizes, /*data=*/{4, 4, 2, 1}),
      out);

  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{0.25, 0.25, 0.5, 1.0}));
}

void OpDivOutTest::test_div_enumerate_a_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_div_enumerate_b_types<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(ENUMERATE_TEST_ENTRY)

  test_div<ScalarType::Bool, ScalarType::Float, ScalarType::Float>();

#undef ENUMERATE_TEST_ENTRY
}

class OpDivScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_div_scalar_out(const Tensor& a, const Scalar& b, Tensor& out) {
    return torch::executor::aten::div_outf(context_, a, b, out);
  }
};

}; // namespace

//
// Correctness Tests
//

/**
 * Uses the function templates above to test all valid combinations of inputs
 * and output dtypes
 */
TEST_F(OpDivOutTest, AllRealDtypesSupported) {
  test_div_enumerate_a_types();
}

TEST_F(OpDivOutTest, BroadcastSupported1) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({2, 1, 2, 1}, {4, 8, 12, 16});
  Tensor b = tf.make({2, 1, 4}, {1, 1, 1, 1, 2, 2, 2, 2});

  // Destination for the broadcasting div. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({2, 2, 2, 4});

  op_div_out(a, b, out);

  Tensor ret = tf.make(
      {2, 2, 2, 4}, {4,  4,  4,  4,  8,  8,  8,  8,  2, 2, 2, 2, 4, 4, 4, 4,
                     12, 12, 12, 12, 16, 16, 16, 16, 6, 6, 6, 6, 8, 8, 8, 8});
  EXPECT_TENSOR_EQ(out, ret);
}

TEST_F(OpDivOutTest, BroadcastSupported2) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({3, 2, 1}, {2, 3, 4, 5, 6, 7});
  Tensor b = tf.make({1, 2, 1}, {2, 2});

  // Destination for the broadcasting div. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({3, 2, 1});

  op_div_out(a, b, out);

  Tensor ret = tf.make({3, 2, 1}, {1, 1.5, 2, 2.5, 3, 3.5});
  EXPECT_TENSOR_EQ(out, ret);
}

TEST_F(OpDivOutTest, BroadcastScalarSupported1) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({2, 1, 3}, {2, 3, 4, 5, 6, 7});
  Tensor b = tf.make({1}, {2});

  // Destination for the broadcasting div. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({2, 1, 3});

  op_div_out(a, b, out);

  Tensor ret = tf.make({2, 1, 3}, {1, 1.5, 2, 2.5, 3, 3.5});
  EXPECT_TENSOR_EQ(out, ret);
}

TEST_F(OpDivOutTest, BroadcastScalarSupported2) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({1, 1, 1}, {8});
  Tensor b = tf.make({3, 1, 1}, {2, 4, 8});

  // Destination for the broadcasting div. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({3, 1, 1});

  op_div_out(a, b, out);

  Tensor ret = tf.make({3, 1, 1}, {4, 2, 1});
  EXPECT_TENSOR_EQ(out, ret);

  std::swap(a, b);
  out = tf.zeros({3, 1, 1});
  op_div_out(a, b, out);
  ret = tf.make({3, 1, 1}, {0.25, 0.5, 1});
  EXPECT_TENSOR_EQ(out, ret);
}

TEST_F(OpDivOutTest, BroadcastScalarRank0Supported) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({1}, {8});
  Tensor b = tf.make({}, {2});

  Tensor out = tf.zeros({1});

  op_div_out(a, b, out);

  Tensor ret = tf.make({1}, {4});
  EXPECT_TENSOR_EQ(out, ret);

  op_div_out(b, a, out);

  ret = tf.make({1}, {0.25});
  EXPECT_TENSOR_EQ(out, ret);
}

TEST_F(OpDivOutTest, BroadcastDimSizeIsOneAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9403896331787109,
       0.33918434381484985,
       0.6973152756690979,
       0.7128887176513672,
       0.9746139049530029,
       0.3507251739501953});
  Tensor y = tf.make({1, 2}, {0.942541241645813, 0.0298004150390625});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.9977172017097473,
       11.381866455078125,
       0.7398247122764587,
       23.922107696533203,
       1.0340278148651123,
       11.769137382507324});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_div_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDivOutTest, BroadcastDimSizeMissingAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9403896331787109,
       0.33918434381484985,
       0.6973152756690979,
       0.7128887176513672,
       0.9746139049530029,
       0.3507251739501953});
  Tensor y = tf.make({2}, {0.942541241645813, 0.0298004150390625});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.9977172017097473,
       11.381866455078125,
       0.7398247122764587,
       23.922107696533203,
       1.0340278148651123,
       11.769137382507324});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_div_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDivOutTest, BroadcastDimSizeIsOneBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.942541241645813, 0.0298004150390625});
  Tensor y = tf.make(
      {3, 2},
      {0.9403896331787109,
       0.33918434381484985,
       0.6973152756690979,
       0.7128887176513672,
       0.9746139049530029,
       0.3507251739501953});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.0022879838943481,
       0.08785904943943024,
       1.351671576499939,
       0.041802339255809784,
       0.9670919179916382,
       0.08496799319982529});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_div_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDivOutTest, BroadcastDimSizeMissingBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.942541241645813, 0.0298004150390625});
  Tensor y = tf.make(
      {3, 2},
      {0.9403896331787109,
       0.33918434381484985,
       0.6973152756690979,
       0.7128887176513672,
       0.9746139049530029,
       0.3507251739501953});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.0022879838943481,
       0.08785904943943024,
       1.351671576499939,
       0.041802339255809784,
       0.9670919179916382,
       0.08496799319982529});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_div_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

//
// Death Tests
//

TEST_F(OpDivOutTest, MismatchedShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor a = tf_int.ones(/*sizes=*/{2});
  Tensor b = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_float.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_div_out(a, b, out));
}

TEST_F(OpDivOutTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_div_invalid_output_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

//
// Dynamic Shape Tests
//

TEST_F(OpDivOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9321315288543701,
       0.013347446918487549,
       0.42016714811325073,
       0.059867143630981445,
       0.951939046382904,
       0.8632845878601074});
  Tensor y = tf.make(
      {3, 2},
      {0.714946985244751,
       0.39985191822052,
       0.9640239477157593,
       0.06885606050491333,
       0.008897960186004639,
       0.468650221824646});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.3037770986557007,
       0.03338097408413887,
       0.4358472228050232,
       0.869453489780426,
       106.98396301269531,
       1.8420659303665161});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_div_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDivOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9321315288543701,
       0.013347446918487549,
       0.42016714811325073,
       0.059867143630981445,
       0.951939046382904,
       0.8632845878601074});
  Tensor y = tf.make(
      {3, 2},
      {0.714946985244751,
       0.39985191822052,
       0.9640239477157593,
       0.06885606050491333,
       0.008897960186004639,
       0.468650221824646});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.3037770986557007,
       0.03338097408413887,
       0.4358472228050232,
       0.869453489780426,
       106.98396301269531,
       1.8420659303665161});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_div_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDivOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9321315288543701,
       0.013347446918487549,
       0.42016714811325073,
       0.059867143630981445,
       0.951939046382904,
       0.8632845878601074});
  Tensor y = tf.make(
      {3, 2},
      {0.714946985244751,
       0.39985191822052,
       0.9640239477157593,
       0.06885606050491333,
       0.008897960186004639,
       0.468650221824646});
  Tensor expected_result = tf.make(
      {3, 2},
      {1.3037770986557007,
       0.03338097408413887,
       0.4358472228050232,
       0.869453489780426,
       106.98396301269531,
       1.8420659303665161});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_div_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDivScalarOutTest, SanityCheckIntScalar) {
  TensorFactory<ScalarType::Int> tf_a;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf_out.zeros(sizes);

  op_div_scalar_out(tf_a.make(sizes, {1, 2, 4, -9}), 2, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {0.5, 1.0, 2.0, -4.5}));
}

TEST_F(OpDivScalarOutTest, SanityCheckFloatScalar) {
  TensorFactory<ScalarType::Int> tf_a;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf_out.zeros(sizes);

  op_div_scalar_out(tf_a.make(sizes, {1, 2, 4, -9}), 2.0, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {0.5, 1.0, 2.0, -4.5}));
}

TEST_F(OpDivScalarOutTest, OptimizedSanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_div_scalar_out(tf.make(sizes, {1.3, 2.1, 4.6, 8.2}), 2.0, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, {0.65, 1.05, 2.3, 4.1}));
}
