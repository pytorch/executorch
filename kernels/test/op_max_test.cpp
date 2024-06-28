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

class OpMaxOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&> op_max_dim_max(
      const Tensor& self,
      int64_t dim,
      bool keepdim,
      Tensor& max,
      Tensor& max_indices) {
    return torch::executor::aten::max_outf(
        context_, self, dim, keepdim, max, max_indices);
  }

  template <ScalarType IN_DTYPE>
  void test_max_out_invalid_dimensions() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<ScalarType::Long> tf_long;

    Tensor self = tf_in.ones(/*sizes=*/{2, 3, 4});
    Tensor max = tf_in.zeros({2, 3, 2});
    Tensor max_indices = tf_in.zeros({2, 3});

    // output tensor dim mismatch
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/true, max, max_indices));

    // output tensor shape incorrect: size of dimension: dim should be 1
    max = tf_in.zeros({2, 3, 2});
    max_indices = tf_in.zeros({2, 3, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/true, max, max_indices));

    // output tensor shape should be squeezed when keepdim is false
    max = tf_in.zeros({2, 3, 1});
    max_indices = tf_in.zeros({2, 3, 1});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/false, max, max_indices));

    // invalid dim
    max = tf_in.zeros({2, 3, 1});
    max_indices = tf_in.zeros({2, 3, 1});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_max_dim_max(self, /*dim=*/3, /*keepdim=*/true, max, max_indices));
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(max_template) */

    TensorFactory<ScalarType::Float> tf;
    TensorFactory<ScalarType::Long> tfl;

    Tensor input = tf.make(
        {2, 3, 4},
        {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
         0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
         0.4900934100151062,   0.8964447379112244,  0.455627977848053,
         0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
         0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
         0.518521785736084,    0.6976675987243652,  0.800011396408081,
         0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
         0.9151939749717712,   0.39709991216659546, 0.8741558790206909});
    Tensor expected_max = tf.make(
        {2, 4},
        {0.49625658988952637,
         0.7682217955589294,
         0.4900934100151062,
         0.8964447379112244,
         0.6976675987243652,
         0.9151939749717712,
         0.39709991216659546,
         0.8741558790206909});
    Tensor expected_max_indices = tfl.make({2, 4}, {0, 0, 1, 1, 1, 2, 2, 2});
    Tensor max = tf.zeros(out_shape, dynamism);
    Tensor max_indices = tfl.zeros(out_shape, dynamism);

    op_max_dim_max(input, 1, false, max, max_indices);
    EXPECT_TENSOR_EQ(max, expected_max);
    EXPECT_TENSOR_EQ(max_indices, expected_max_indices);
  }

  template <ScalarType IN_DTYPE>
  void test_max_out_dtype() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<ScalarType::Long> tf_long;
    // clang-format off
    Tensor self = tf_in.make(
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

    Tensor max = tf_in.zeros({2, 4});
    Tensor max_indices = tf_long.zeros({2, 4});
    op_max_dim_max(self, /*dim=*/1, /*keepdim=*/false, max, max_indices);
    // clang-format off
    EXPECT_TENSOR_CLOSE(max, tf_in.make(
      {2, 4},
      {
        4, 2, 4, 4,
        4, 2, 4, 4
      }));

    EXPECT_TENSOR_EQ(max_indices, tf_long.make(
      {2, 4},
      {
        1, 1, 2, 0,
        0, 0, 2, 1
      }));
    // clang-format on

    // negative dim should work
    op_max_dim_max(self, /*dim=*/-2, /*keepdim=*/false, max, max_indices);
    // clang-format off
    EXPECT_TENSOR_CLOSE(max, tf_in.make(
      {2, 4},
      {
        4, 2, 4, 4,
        4, 2, 4, 4
      }));
    EXPECT_TENSOR_EQ(max_indices, tf_long.make(
      {2, 4},
      {
        1, 1, 2, 0,
        0, 0, 2, 1
      }));
    // clang-format on

    // keepdim should work
    max = tf_in.zeros({2, 3, 1});
    max_indices = tf_long.zeros({2, 3, 1});
    op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/true, max, max_indices);
    EXPECT_TENSOR_CLOSE(max, tf_in.make({2, 3, 1}, {4, 4, 4, 4, 4, 4}));
    EXPECT_TENSOR_EQ(max_indices, tf_long.make({2, 3, 1}, {3, 0, 2, 0, 3, 2}));
  }
};

template <>
void OpMaxOutTest::test_max_out_dtype<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
  Tensor self = tf_bool.make(
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

  Tensor max = tf_bool.zeros({2, 3, 1});
  Tensor max_indices = tf_long.zeros({2, 3, 1});

  // +/-inf and nan should work
  op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/true, max, max_indices);
  // clang-format off
  EXPECT_TENSOR_CLOSE(
      max, tf_bool.make(
        {2, 3, 1},
        {
          true,
          false,
          true,

          true,
          true,
          true
        }));
  EXPECT_TENSOR_EQ(max_indices, tf_long.make(
    {2, 3, 1},
    {
      0,
      0,
      1,

      2,
      3,
      0
    }));
  // clang-format on
}

TEST_F(OpMaxOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_max_out_invalid_dimensions<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpMaxOutTest, MismatchedDTypesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf_float.ones(/*sizes=*/{2, 3, 4});
  Tensor max = tf_long.zeros({2, 3, 1});
  Tensor max_indices = tf_long.zeros({2, 3, 1});

  // dtype of self and max should match
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/true, max, max_indices));

  // max_value tensor should have long as dtype
  max = tf_float.zeros({2, 3, 1});
  max_indices = tf_float.zeros({2, 3, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/true, max, max_indices));
}

TEST_F(OpMaxOutTest, AllRealInputLongOutputPasses) {
#define TEST_ENTRY(ctype, dtype) test_max_out_dtype<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpMaxOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
  Tensor self = tf_float.make(
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

  Tensor max = tf_float.zeros({2, 3, 1});
  Tensor max_indices = tf_long.zeros({2, 3, 1});

  // +/-inf and nan should work
  op_max_dim_max(self, /*dim=*/-1, /*keepdim=*/true, max, max_indices);
  EXPECT_TENSOR_CLOSE(
      max, tf_float.make({2, 3, 1}, {INFINITY, INFINITY, NAN, NAN, NAN, NAN}));
  // clang-format off
  EXPECT_TENSOR_EQ(max_indices, tf_long.make(
    {2, 3, 1},
    {
      3,
      0,
      0,

      0,
      2,
      1
    }));
  // clang-format on
}

/* %python
import torch
torch.manual_seed(0)
input = torch.rand(2, 3, 4)
dim = 1
keepdim = False
(values, indices) = torch.max(input, dim, keepdim=keepdim)

max_template = f"""
  {declare_tensor_factory("ScalarType::Float", "tf")}
  {declare_tensor_factory("ScalarType::Long", "tfl")}

  {declare_tensor_make_t("input", "tf")}
  {declare_tensor_make_t("values", "tf", "expected_max")}
  {declare_tensor_make_t("indices", "tfl", "expected_max_indices")}
  {declare_tensor_zeros("out_shape, dynamism", "tf", "max")}
  {declare_tensor_zeros("out_shape, dynamism", "tfl", "max_indices")}

  op_max_dim_max(input, $dim$, $keepdim$, max, max_indices);
  EXPECT_TENSOR_EQ(max, expected_max);
  EXPECT_TENSOR_EQ(max_indices, expected_max_indices);""" */

TEST_F(OpMaxOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpMaxOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpMaxOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
