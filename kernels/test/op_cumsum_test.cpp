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
#include <cmath>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpCumSumOutTest : public OperatorTest {
 protected:
  Tensor& op_cumsum_out(
      const Tensor& self,
      int64_t dim,
      optional<ScalarType> enforced_dtype,
      Tensor& out) {
    return torch::executor::aten::cumsum_outf(
        context_, self, dim, enforced_dtype, out);
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_cumsum_out_dtype() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;
    // clang-format off
    Tensor in = tf_in.make(
      {2, 4},
      {
        0, 1,  2,  4,
        8, 16, 32, 64
      });
    // clang-format on

    Tensor out = tf_out.zeros({2, 4});
    optional<ScalarType> enforced_dtype = OUT_DTYPE;
    op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);

    // clang-format off
    Tensor expected = tf_out.make(
      {2, 4},
      {
        0, 1,  3,  7,
        8, 24, 56, 120
      });
    // clang-format on

    EXPECT_TENSOR_CLOSE(out, expected);

    // negative dim should work
    op_cumsum_out(in, /*dim=*/-1, enforced_dtype, out);
    EXPECT_TENSOR_CLOSE(out, expected);

    op_cumsum_out(in, /*dim=*/0, enforced_dtype, out);
    // clang-format off
    expected = tf_out.make(
      {2, 4},
      {
        0, 1,  2,  4,
        8, 17, 34, 68
      });
    // clang-format on
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType OUT_DTYPE>
  void test_cumsum_out_float() {
    TensorFactory<ScalarType::Float> tf_float;
    TensorFactory<OUT_DTYPE> tf_out;

    Tensor in = tf_float.make({1, 2}, {1, INFINITY});
    Tensor out = tf_out.zeros({1, 2});
    optional<ScalarType> enforced_dtype = OUT_DTYPE;
    op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 2}, {1, INFINITY}));

    in = tf_float.make({1, 2}, {1, -INFINITY});
    op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 2}, {1, -INFINITY}));

    in = tf_float.make({1, 2}, {1, NAN});
    op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 2}, {1, NAN}));

    in = tf_float.make({1, 2}, {-INFINITY, INFINITY});
    op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 2}, {-INFINITY, NAN}));
  }
};

TEST_F(OpCumSumOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Float> tff;

  Tensor in = tff.make({1, 3}, {0, 1, 2});

  // Output shape should be (1, 3)
  Tensor out = tff.zeros({1, 3});

  // Dim out of bounds
  optional<ScalarType> enforced_dtype;
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_cumsum_out(in, /*dim=*/3, enforced_dtype, out));

  // wrong_out has incompatible dim
  Tensor wrong_out = tff.zeros({2, 10, 4});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_cumsum_out(in, /*dim=*/1, enforced_dtype, wrong_out));
}

/* A generic smoke test that works for the supported dtypes with
 * enforced_dtype specified.
 */
TEST_F(OpCumSumOutTest, EnforcedDtypePasses) {
// Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_cumsum_out_dtype<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_REAL_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpCumSumOutTest, TypeCastCornerCases) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Byte> tf_byte;

  // Cast floating point to int
  Tensor in = tf_float.make({1, 2}, {1.1, 2.2});
  Tensor out = tf_int.zeros({1, 2});
  optional<ScalarType> enforced_dtype = ScalarType::Int;
  op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf_int.make({1, 2}, {1, 3}));

  // Cast negative values to unsigned type
  in = tf_int.make({1, 2}, {-1, -2});
  out = tf_byte.zeros({1, 2});
  enforced_dtype = ScalarType::Byte;
  op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf_byte.make({1, 2}, {255, 253}));

  // Cast negative float values to int, float should rounding toward zero
  in = tf_float.make({1, 2}, {-1.9, -2.9});
  out = tf_int.zeros({1, 2});
  enforced_dtype = ScalarType::Int;
  op_cumsum_out(in, /*dim=*/1, enforced_dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf_int.make({1, 2}, {-1, -3}));
}

/* A generic smoke test that works for the supported dtypes with
 * enforced_dtype specified.
 */
TEST_F(OpCumSumOutTest, FloatSpecificTest) {
// Float/double specific +/-Inf and NAN test
#define TEST_ENTRY_FLOAT_SPECIFIC_CASES(ctype, dtype) \
  test_cumsum_out_float<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY_FLOAT_SPECIFIC_CASES);
#undef TEST_ENTRY_FLOAT_SPECIFIC_CASES
}

TEST_F(OpCumSumOutTest, SimpleGeneratedCase) {
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
      {1.0,  2.0,  3.0, 4.0,  5.0,  6.0, 7.0,  8.0,  9.0, 10.0, 1.0,  2.0,  3.0,
       4.0,  5.0,  6.0, 7.0,  8.0,  9.0, 10.0, 1.0,  2.0, 3.0,  4.0,  5.0,  6.0,
       7.0,  8.0,  9.0, 10.0, 1.0,  2.0, 3.0,  4.0,  5.0, 6.0,  7.0,  8.0,  9.0,
       10.0, 1.0,  2.0, 3.0,  4.0,  5.0, 6.0,  7.0,  8.0, 9.0,  10.0, 1.0,  2.0,
       3.0,  4.0,  5.0, 6.0,  7.0,  8.0, 9.0,  10.0, 1.0, 2.0,  3.0,  4.0,  5.0,
       6.0,  7.0,  8.0, 9.0,  10.0, 1.0, 2.0,  3.0,  4.0, 5.0,  6.0,  7.0,  8.0,
       9.0,  10.0, 1.0, 2.0,  3.0,  4.0, 5.0,  6.0,  7.0, 8.0,  9.0,  10.0, 1.0,
       2.0,  3.0,  4.0, 5.0,  6.0,  7.0, 8.0,  9.0,  10.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_cumsum_out(x, 1, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpCumSumOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.8651182055473328,
       0.44230276346206665,
       0.7190993428230286,
       0.8998266458511353,
       0.9957790374755859});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_cumsum_out(x, 1, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpCumSumOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.8651182055473328,
       0.44230276346206665,
       0.7190993428230286,
       0.8998266458511353,
       0.9957790374755859});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_cumsum_out(x, 1, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpCumSumOutTest, DynamicShapeUnbound) {
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
  Tensor expected_result = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.8651182055473328,
       0.44230276346206665,
       0.7190993428230286,
       0.8998266458511353,
       0.9957790374755859});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_cumsum_out(x, 1, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
