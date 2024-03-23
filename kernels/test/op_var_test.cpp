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
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpVarOutTest : public OperatorTest {
 protected:
  Tensor& op_var_out(
      const Tensor& self,
      optional<ArrayRef<int64_t>> dim,
      bool unbiased,
      bool keepdim,
      Tensor& out) {
    return torch::executor::aten::var_outf(
        context_, self, dim, unbiased, keepdim, out);
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_var_out_invalid_dimensions() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;

    // clang-format off
    Tensor self = tf_in.make(
      {2, 3, 4},
      {
        0, 1, 2,  3,
        4, 5, 6,  7,
        8, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
      });
    // clang-format on
    Tensor out = tf_out.zeros({2, 3, 1});
    optional<ScalarType> dtype = OUT_DTYPE;

    // out-of-bound dim in dim list
    int64_t dims_1[1] = {3};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_var_out(
            self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/true, out));

    // the same dim appears multiple times in list of dims
    int64_t dims_2[2] = {2, 2};
    optional_dim_list = ArrayRef<int64_t>{dims_2, 2};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_var_out(
            self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/true, out));
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_var_out_invalid_shape() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;

    // clang-format off
    Tensor self = tf_in.make(
      {2, 3, 4},
      {
        0, 1, 2,  3,
        4, 5, 6,  7,
        8, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
      });
    // clang-format on

    // dimension size mismatch when keepdim is true
    Tensor out = tf_out.zeros({2, 4});
    optional<ScalarType> dtype = OUT_DTYPE;
    int64_t dims_1[1] = {1};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_var_out(
            self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/true, out));

    // dimension size mismatch when keepdim is false
    out = tf_out.zeros({2, 1, 4});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_var_out(
            self,
            optional_dim_list,
            /*unbiased=*/true,
            /*keepdim=*/false,
            out));
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_var_out_dtype() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;
    // clang-format off
    Tensor self = tf_in.make(
      {2, 3, 4},
      {
        0, 1, 2,  3,
        4, 5, 6,  7,
        8, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
      });
    // clang-format on

    // keepdim=true should work
    Tensor out = tf_out.zeros({2, 3, 1});
    int64_t dims_1[1] = {2};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
    optional<ScalarType> dtype = OUT_DTYPE;
    op_var_out(
        self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/true, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 3, 1},
      {
        1.666667,
        1.666667,
        1.666667,

        1.666667,
        1.666667,
        1.666667
      }));
    // clang-format on

    // keepdim=false should work
    out = tf_out.zeros({2, 3});
    op_var_out(
        self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/false, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 3},
      {
        1.666667, 1.666667, 1.666667,
        1.666667, 1.666667, 1.666667
      }));
    // clang-format on

    // dim list with multiple dimensions should work
    out = tf_out.zeros({1, 1, 4});
    int64_t dims_2[2] = {0, 1};
    optional_dim_list = ArrayRef<int64_t>{dims_2, 2};
    op_var_out(
        self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/true, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 4}, {56.0, 56.0, 56.0, 56.0}));

    out = tf_out.zeros({4});
    op_var_out(
        self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/false, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({4}, {56.0, 56.0, 56.0, 56.0}));

    // dim list with negative dimensions should work
    out = tf_out.zeros({2, 1, 4});
    int64_t dims_3[1] = {-2};
    optional_dim_list = ArrayRef<int64_t>{dims_3, 1};
    op_var_out(
        self, optional_dim_list, /*unbiased=*/false, /*keepdim=*/true, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 1, 4},
      {
        10.666667, 10.666667, 10.666667, 10.666667,

        10.666667, 10.666667, 10.666667, 10.666667,
      }));
    // clang-format on

    // empty/null dim list should work
    out = tf_out.zeros({1, 1, 1});
    optional<ArrayRef<int64_t>> null_dim_list;
    op_var_out(self, null_dim_list, /*unbiased=*/true, /*keepdim=*/true, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 1}, {50.0}));

    optional<ArrayRef<int64_t>> empty_dim_list{ArrayRef<int64_t>{}};
    op_var_out(self, empty_dim_list, /*unbiased=*/false, /*keepdim=*/true, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 1}, {47.916668}));

    out = tf_out.zeros({});
    op_var_out(self, null_dim_list, /*unbiased=*/false, /*keepdim=*/false, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({}, {47.916668}));

    op_var_out(self, empty_dim_list, /*unbiased=*/true, /*keepdim=*/false, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({}, {50.0}));
  }
};

class OpVarCorrectionOutTest : public OperatorTest {
 protected:
  Tensor& op_var_correction_out(
      const Tensor& self,
      optional<ArrayRef<int64_t>> dim,
      optional<Scalar>& correction,
      bool keepdim,
      Tensor& out) {
    return torch::executor::aten::var_outf(
        context_, self, dim, correction, keepdim, out);
  }
};

TEST_F(OpVarOutTest, InvalidDimensionListDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_var_out_invalid_dimensions<                                        \
      ScalarType::INPUT_DTYPE,                                            \
      ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOAT_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarOutTest, InvalidShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_var_out_invalid_shape<                                             \
      ScalarType::INPUT_DTYPE,                                            \
      ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOAT_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarOutTest, InvalidDTypeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Int> tf_int;

  // clang-format off
  Tensor self = tf_int.make(
    {2, 3, 4},
    {
      0, 1, 2,  3,
      4, 5, 6,  7,
      8, 9, 10, 11,

      12, 13, 14, 15,
      16, 17, 18, 19,
      20, 21, 22, 23,
    });
  // clang-format on

  // keepdim=true should work
  Tensor out = tf_float.zeros({2, 3, 1});
  int64_t dims_1[1] = {2};
  optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_var_out(
          self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/true, out));
}

TEST_F(OpVarOutTest, AllFloatInputFloatOutputPasses) {
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_var_out_dtype<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOAT_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Float> tf_float;
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

  Tensor out = tf_float.zeros({2, 3, 1});
  int64_t dims[1] = {-1};
  optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims, 1}};
  optional<ScalarType> dtype;
  op_var_out(self, optional_dim_list, /*unbiased=*/true, /*keepdim=*/true, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(out, tf_float.make(
    {2, 3, 1},
    {
      NAN,
      NAN,
      NAN,

      NAN,
      NAN,
      NAN
    }));
  // clang-format on
}

TEST_F(OpVarOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({3, 2}, {0.49, 0.40, 0.56, 0.38, 0.49, 0.56});
  Tensor expected_result = tf.make({3}, {0.004050, 0.016200, 0.002450});

  Tensor out =
      tf.zeros({3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_var_out(
      x, ArrayRef<int64_t>{1}, /*unbiased=*/true, /*keepdim=*/false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpVarOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({3, 2}, {0.49, 0.40, 0.56, 0.38, 0.49, 0.56});
  Tensor expected_result = tf.make({3}, {0.004050, 0.016200, 0.002450});

  Tensor out =
      tf.zeros({10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_var_out(
      x, ArrayRef<int64_t>{1}, /*unbiased=*/true, /*keepdim=*/false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpVarOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({3, 2}, {0.49, 0.40, 0.56, 0.38, 0.49, 0.56});
  Tensor expected_result = tf.make({3}, {0.004050, 0.016200, 0.002450});

  Tensor out =
      tf.zeros({1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_var_out(
      x, ArrayRef<int64_t>{1}, /*unbiased=*/true, /*keepdim=*/false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpVarCorrectionOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({2, 3}, {4.9, 4.0, 5.6, 3.8, 4.9, 5.6});
  Tensor expected = tf.make({2}, {0.72693, 0.93032});
  optional<Scalar> correction(1.23);
  Tensor out = tf.zeros({2});

  op_var_correction_out(
      x, ArrayRef<int64_t>{1}, correction, /*keepdim=*/false, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}
