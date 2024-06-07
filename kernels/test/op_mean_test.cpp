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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpMeanOutTest : public OperatorTest {
 protected:
  Tensor& op_mean_out(
      const Tensor& self,
      optional<ArrayRef<int64_t>> dim,
      bool keepdim,
      optional<ScalarType> dtype,
      Tensor& out) {
    return torch::executor::aten::mean_outf(
        context_, self, dim, keepdim, dtype, out);
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_mean_dim_out_invalid_dimensions() {
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
        op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out));

    // the same dim appears multiple times in list of dims
    int64_t dims_2[2] = {2, 2};
    optional_dim_list = ArrayRef<int64_t>{dims_2, 2};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out));
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_mean_dim_out_invalid_shape() {
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
        op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out));

    // dimension size mismatch when keepdim is false
    out = tf_out.zeros({2, 1, 4});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_mean_out(self, optional_dim_list, /*keepdim=*/false, dtype, out));
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_mean_dim_out_dtype() {
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
    op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 3, 1},
      {
        1.5,
        5.5,
        9.5,

        13.5,
        17.5,
        21.5
      }));
    // clang-format on

    // keepdim=false should work
    out = tf_out.zeros({2, 3});
    op_mean_out(self, optional_dim_list, /*keepdim=*/false, dtype, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 3},
      {
        1.5,  5.5,  9.5,
        13.5, 17.5, 21.5
      }));
    // clang-format on

    // dim list with multiple dimensions should work
    out = tf_out.zeros({1, 1, 4});
    int64_t dims_2[2] = {0, 1};
    optional_dim_list = ArrayRef<int64_t>{dims_2, 2};
    op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 4}, {10, 11, 12, 13}));

    out = tf_out.zeros({4});
    op_mean_out(self, optional_dim_list, false, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({4}, {10, 11, 12, 13}));

    // dim list with negative dimensions should work
    out = tf_out.zeros({2, 1, 4});
    int64_t dims_3[1] = {-2};
    optional_dim_list = ArrayRef<int64_t>{dims_3, 1};
    op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 1, 4},
      {
        4,  5,  6,  7,

        16, 17, 18, 19,
      }));
    // clang-format on

    // empty/null dim list should work
    out = tf_out.zeros({1, 1, 1});
    optional<ArrayRef<int64_t>> null_dim_list;
    op_mean_out(self, null_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 1}, {11.5}));

    optional<ArrayRef<int64_t>> empty_dim_list{ArrayRef<int64_t>{}};
    op_mean_out(self, empty_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 1}, {11.5}));

    out = tf_out.zeros({});
    op_mean_out(self, null_dim_list, /*keepdim=*/false, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({}, {11.5}));

    op_mean_out(self, empty_dim_list, /*keepdim=*/false, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({}, {11.5}));
  }

  template <ScalarType OUT_DTYPE>
  void test_mean_dim_out_bool() {
    TensorFactory<ScalarType::Bool> tf_bool;
    TensorFactory<OUT_DTYPE> tf_float;
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

    Tensor out = tf_float.zeros({1, 1, 4});
    int64_t dims[2] = {0, 1};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims, 2}};
    optional<ScalarType> dtype = OUT_DTYPE;
    op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(
        out,
        tf_float.make({1, 1, 4}, {0.333333, 0.333333, 0.666667, 0.333333}));
  }
};

template <>
void OpMeanOutTest::
    test_mean_dim_out_dtype<ScalarType::Bool, ScalarType::Float>() {
  test_mean_dim_out_bool<ScalarType::Float>();
}

template <>
void OpMeanOutTest::
    test_mean_dim_out_dtype<ScalarType::Bool, ScalarType::Double>() {
  test_mean_dim_out_bool<ScalarType::Double>();
}

TEST_F(OpMeanOutTest, InvalidDimensionListDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_mean_dim_out_invalid_dimensions<                                   \
      ScalarType::INPUT_DTYPE,                                            \
      ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOAT_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpMeanOutTest, InvalidShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_mean_dim_out_invalid_shape<                                        \
      ScalarType::INPUT_DTYPE,                                            \
      ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOAT_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpMeanOutTest, MismatchedDTypesDies) {
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
  optional<ScalarType> dtype;

  // self tensor must have a floating point dtype when dtype is not specified
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out));

  dtype = ScalarType::Double;
  // out tensor should be of the same dtype with dtype when dtype is specified
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out));
}

TEST_F(OpMeanOutTest, AllRealInputFloatOutputPasses) {
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_mean_dim_out_dtype<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOAT_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpMeanOutTest, HalfSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_mean_dim_out_dtype<ScalarType::dtype, ScalarType::Half>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY

#define TEST_ENTRY(ctype, dtype) \
  test_mean_dim_out_dtype<ScalarType::Half, ScalarType::dtype>();
  ET_FORALL_FLOATH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpMeanOutTest, InfinityAndNANTest) {
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
  op_mean_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(out, tf_float.make(
    {2, 3, 1},
    {
      INFINITY,
      NAN,
      NAN,

      NAN,
      NAN,
      NAN
    }));
  // clang-format on
}

TEST_F(OpMeanOutTest, SimpleGeneratedCase) {
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
  Tensor expected_result =
      tf.make({10}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  Tensor out = tf.zeros({10});
  Tensor ret =
      op_mean_out(x, ArrayRef<int64_t>{1}, false, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMeanOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49627798795700073,
       0.40115922689437866,
       0.5627331733703613,
       0.3858276605606079,
       0.4964867830276489,
       0.5637965202331543});
  Tensor expected_result = tf.make(
      {3}, {0.4487186074256897, 0.4742804169654846, 0.5301416516304016});

  Tensor out =
      tf.zeros({3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret =
      op_mean_out(x, ArrayRef<int64_t>{1}, false, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMeanOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49627798795700073,
       0.40115922689437866,
       0.5627331733703613,
       0.3858276605606079,
       0.4964867830276489,
       0.5637965202331543});
  Tensor expected_result = tf.make(
      {3}, {0.4487186074256897, 0.4742804169654846, 0.5301416516304016});

  Tensor out =
      tf.zeros({10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret =
      op_mean_out(x, ArrayRef<int64_t>{1}, false, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMeanOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49627798795700073,
       0.40115922689437866,
       0.5627331733703613,
       0.3858276605606079,
       0.4964867830276489,
       0.5637965202331543});
  Tensor expected_result = tf.make(
      {3}, {0.4487186074256897, 0.4742804169654846, 0.5301416516304016});

  Tensor out =
      tf.zeros({1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret =
      op_mean_out(x, ArrayRef<int64_t>{1}, false, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
