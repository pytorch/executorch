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
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpMinOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&> op_min_dim_min(
      const Tensor& in,
      int64_t dim,
      bool keepdim,
      Tensor& min,
      Tensor& min_indices) {
    return torch::executor::aten::min_outf(
        context_, in, dim, keepdim, min, min_indices);
  }

  template <ScalarType IN_DTYPE>
  void test_min_out_invalid_dimensions() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<ScalarType::Long> tf_long;

    Tensor in = tf_in.ones(/*sizes=*/{2, 3, 4});
    Tensor min = tf_in.zeros({2, 3, 2});
    Tensor min_indices = tf_in.zeros({2, 3});

    // output tensor dim mismatch
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/true, min, min_indices));

    // output tensor shape incorrect: size of dimension: dim should be 1
    min = tf_in.zeros({2, 3, 2});
    min_indices = tf_in.zeros({2, 3, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/true, min, min_indices));

    // output tensor shape should be squeezed when keepdim is false
    min = tf_in.zeros({2, 3, 1});
    min_indices = tf_in.zeros({2, 3, 1});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/false, min, min_indices));

    // invalid dim
    min = tf_in.zeros({2, 3, 1});
    min_indices = tf_in.zeros({2, 3, 1});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_min_dim_min(in, /*dim=*/3, /*keepdim=*/true, min, min_indices));
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(min_template) */

    TensorFactory<ScalarType::Float> tf;
    TensorFactory<ScalarType::Long> tfl;

    // clang-format off
  Tensor input = tf.make(
      {2, 3, 4},
      {0.49, 0.76, 0.08, 0.13,
       0.30, 0.63, 0.49, 0.89,
       0.45, 0.63, 0.34, 0.40,

       0.02, 0.16, 0.29, 0.51,
       0.69, 0.80, 0.16, 0.28,
       0.68, 0.91, 0.39, 0.87});
  Tensor expected_min = tf.make(
      {2, 4},
      {0.30, 0.63, 0.08, 0.13,
       0.02, 0.16, 0.16, 0.28});
    // clang-format on

    Tensor expected_min_indices = tfl.make({2, 4}, {1, 1, 0, 0, 0, 0, 1, 1});
    Tensor min = tf.zeros(out_shape, dynamism);
    Tensor min_indices = tfl.zeros(out_shape, dynamism);

    op_min_dim_min(input, 1, false, min, min_indices);
    EXPECT_TENSOR_EQ(min, expected_min);
    EXPECT_TENSOR_EQ(min_indices, expected_min_indices);
  }

  template <ScalarType IN_DTYPE>
  void test_min_out_dtype() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<ScalarType::Long> tf_long;
    // clang-format off
  Tensor in = tf_in.make(
    {2, 3, 4},
    {
      0, 1, 2, 4,
      4, 2, 1, 0,
      1, 0, 4, 2,

      4, 2, 1, 0,
      0, 1, 2, 4,
      1, 0, 4, 2,
    });
    // clang-format on

    Tensor min = tf_in.zeros({2, 4});
    Tensor min_indices = tf_long.zeros({2, 4});
    op_min_dim_min(in, /*dim=*/1, /*keepdim=*/false, min, min_indices);
    // clang-format off
  EXPECT_TENSOR_CLOSE(min, tf_in.make(
    {2, 4},
    {
      0, 0, 1, 0,
      0, 0, 1, 0
    }));

  EXPECT_TENSOR_EQ(min_indices, tf_long.make(
    {2, 4},
    {
      0, 2, 1, 1,
      1, 2, 0, 0
    }));
    // clang-format on

    // negative dim should work
    op_min_dim_min(in, /*dim=*/-2, /*keepdim=*/false, min, min_indices);
    // clang-format off
  EXPECT_TENSOR_CLOSE(min, tf_in.make(
    {2, 4},
    {
      0, 0, 1, 0,
      0, 0, 1, 0
    }));
  EXPECT_TENSOR_EQ(min_indices, tf_long.make(
    {2, 4},
    {
      0, 2, 1, 1,
      1, 2, 0, 0
    }));
    // clang-format on

    // keepdim should work
    min = tf_in.zeros({2, 3, 1});
    min_indices = tf_long.zeros({2, 3, 1});
    op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/true, min, min_indices);
    EXPECT_TENSOR_CLOSE(min, tf_in.make({2, 3, 1}, {0, 0, 0, 0, 0, 0}));
    EXPECT_TENSOR_EQ(min_indices, tf_long.make({2, 3, 1}, {0, 3, 1, 3, 0, 1}));
  }
};

template <>
void OpMinOutTest::test_min_out_dtype<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
Tensor in = tf_bool.make(
  {2, 3, 4},
  {
    true,  false, true,  false,
    false, false, false, false,
    false, true,  true,  false,

    false, false, true,  false,
    false, false, false, true,
    true,  true,  true,  true,
  });
  // clang-format on

  Tensor min = tf_bool.zeros({2, 3, 1});
  Tensor min_indices = tf_long.zeros({2, 3, 1});

  // +/-inf and nan should work
  op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/true, min, min_indices);
  // clang-format off
EXPECT_TENSOR_CLOSE(
    min, tf_bool.make(
      {2, 3, 1},
      {
        false,
        false,
        false,

        false,
        false,
        true
      }));
EXPECT_TENSOR_EQ(min_indices, tf_long.make(
  {2, 3, 1},
  {
    1,
    0,
    0,

    0,
    0,
    0
  }));
  // clang-format on
}

TEST_F(OpMinOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_min_out_invalid_dimensions<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpMinOutTest, MismatchedDTypesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor in = tf_float.ones(/*sizes=*/{2, 3, 4});
  Tensor min = tf_long.zeros({2, 3, 1});
  Tensor min_indices = tf_long.zeros({2, 3, 1});

  // dtype of in and min should match
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/true, min, min_indices));

  // min_value tensor should have long as dtype
  min = tf_float.zeros({2, 3, 1});
  min_indices = tf_float.zeros({2, 3, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/true, min, min_indices));
}

TEST_F(OpMinOutTest, AllRealInputLongOutputPasses) {
#define TEST_ENTRY(ctype, dtype) test_min_out_dtype<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpMinOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
  Tensor in = tf_float.make(
    {2, 3, 4},
    {
      0,        1,         2,        INFINITY,
      INFINITY, -INFINITY, 1,        0,
      NAN,      INFINITY, -INFINITY, 2,

      NAN, NAN,      1,    0,
      0,   INFINITY, NAN,  4,
      1,   NAN,      3.14, 2,
    });
  // clang-format on

  Tensor min = tf_float.zeros({2, 3, 1});
  Tensor min_indices = tf_long.zeros({2, 3, 1});

  // +/-inf and nan should work
  op_min_dim_min(in, /*dim=*/-1, /*keepdim=*/true, min, min_indices);
  EXPECT_TENSOR_CLOSE(
      min, tf_float.make({2, 3, 1}, {0, -INFINITY, NAN, NAN, NAN, NAN}));
  // clang-format off
  EXPECT_TENSOR_EQ(min_indices, tf_long.make(
    {2, 3, 1},
    {
      0,
      1,
      0,

      0,
      2,
      1
    }));
  // clang-format on
}

TEST_F(OpMinOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpMinOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpMinOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
