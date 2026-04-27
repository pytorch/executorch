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
using executorch::aten::ArrayRef;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

namespace {
void expect_tensor_close_with_increased_tol(
    const Tensor& actual,
    const Tensor& expected) {
  if (actual.scalar_type() == ScalarType::BFloat16 ||
      actual.scalar_type() == ScalarType::Half) {
    EXPECT_TENSOR_CLOSE_WITH_TOL(expected, actual, 1e-2, 1e-2);
  } else {
    EXPECT_TENSOR_CLOSE(expected, actual);
  }
}
} // namespace

class OpVarMeanCorrectionOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&> op_var_mean_correction_out(
      const Tensor& self,
      optional<ArrayRef<int64_t>> dim,
      optional<Scalar>& correction,
      bool keepdim,
      Tensor& out0,
      Tensor& out1) {
    return torch::executor::aten::var_mean_outf(
        context_, self, dim, correction, keepdim, out0, out1);
  }

  template <ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    Tensor x = tf.make({2, 3}, {4.9, 4.0, 5.6, 3.8, 4.9, 5.6});
    Tensor expected_var = tf.make({2}, {0.72693, 0.93032});
    Tensor expected_mean = tf.make({2}, {4.833333, 4.766667});
    optional<Scalar> correction(1.23);
    Tensor var_out = tf.zeros({2});
    Tensor mean_out = tf.zeros({2});

    op_var_mean_correction_out(
        x,
        ArrayRef<int64_t>{1},
        correction,
        /*keepdim=*/false,
        var_out,
        mean_out);
    expect_tensor_close_with_increased_tol(var_out, expected_var);
    expect_tensor_close_with_increased_tol(mean_out, expected_mean);
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_keepdim() {
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

    // keepdim=true
    Tensor var_out = tf_out.zeros({2, 3, 1});
    Tensor mean_out = tf_out.zeros({2, 3, 1});
    int64_t dims_1[1] = {2};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
    optional<Scalar> correction(1);
    op_var_mean_correction_out(
        self,
        optional_dim_list,
        correction,
        /*keepdim=*/true,
        var_out,
        mean_out);
    // clang-format off
    expect_tensor_close_with_increased_tol(var_out, tf_out.make(
      {2, 3, 1},
      {
        1.666667,
        1.666667,
        1.666667,

        1.666667,
        1.666667,
        1.666667,
      }));
    expect_tensor_close_with_increased_tol(mean_out, tf_out.make(
      {2, 3, 1},
      {
        1.5,
        5.5,
        9.5,

        13.5,
        17.5,
        21.5,
      }));
    // clang-format on

    // keepdim=false
    var_out = tf_out.zeros({2, 3});
    mean_out = tf_out.zeros({2, 3});
    op_var_mean_correction_out(
        self,
        optional_dim_list,
        correction,
        /*keepdim=*/false,
        var_out,
        mean_out);
    // clang-format off
    expect_tensor_close_with_increased_tol(var_out, tf_out.make(
      {2, 3},
      {
        1.666667, 1.666667, 1.666667,
        1.666667, 1.666667, 1.666667,
      }));
    expect_tensor_close_with_increased_tol(mean_out, tf_out.make(
      {2, 3},
      {
        1.5, 5.5, 9.5,
        13.5, 17.5, 21.5,
      }));
    // clang-format on
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_multiple_dims() {
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

    Tensor var_out = tf_out.zeros({1, 1, 4});
    Tensor mean_out = tf_out.zeros({1, 1, 4});
    int64_t dims[2] = {0, 1};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims, 2}};
    optional<Scalar> correction(1);
    op_var_mean_correction_out(
        self,
        optional_dim_list,
        correction,
        /*keepdim=*/true,
        var_out,
        mean_out);
    expect_tensor_close_with_increased_tol(
        var_out, tf_out.make({1, 1, 4}, {56.0, 56.0, 56.0, 56.0}));
    expect_tensor_close_with_increased_tol(
        mean_out, tf_out.make({1, 1, 4}, {10.0, 11.0, 12.0, 13.0}));

    var_out = tf_out.zeros({4});
    mean_out = tf_out.zeros({4});
    op_var_mean_correction_out(
        self,
        optional_dim_list,
        correction,
        /*keepdim=*/false,
        var_out,
        mean_out);
    expect_tensor_close_with_increased_tol(
        var_out, tf_out.make({4}, {56.0, 56.0, 56.0, 56.0}));
    expect_tensor_close_with_increased_tol(
        mean_out, tf_out.make({4}, {10.0, 11.0, 12.0, 13.0}));
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_negative_dim() {
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

    Tensor var_out = tf_out.zeros({2, 1, 4});
    Tensor mean_out = tf_out.zeros({2, 1, 4});
    int64_t dims[1] = {-2};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims, 1}};
    optional<Scalar> correction(0);
    op_var_mean_correction_out(
        self,
        optional_dim_list,
        correction,
        /*keepdim=*/true,
        var_out,
        mean_out);
    // clang-format off
    expect_tensor_close_with_increased_tol(var_out, tf_out.make(
      {2, 1, 4},
      {
        10.666667, 10.666667, 10.666667, 10.666667,

        10.666667, 10.666667, 10.666667, 10.666667,
      }));
    expect_tensor_close_with_increased_tol(mean_out, tf_out.make(
      {2, 1, 4},
      {
        4.0, 5.0, 6.0, 7.0,

        16.0, 17.0, 18.0, 19.0,
      }));
    // clang-format on
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_null_and_empty_dim_list() {
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

    // null dim list, correction=1 (unbiased), keepdim=true
    Tensor var_out = tf_out.zeros({1, 1, 1});
    Tensor mean_out = tf_out.zeros({1, 1, 1});
    optional<ArrayRef<int64_t>> null_dim_list;
    optional<Scalar> correction(1);
    op_var_mean_correction_out(
        self, null_dim_list, correction, /*keepdim=*/true, var_out, mean_out);
    expect_tensor_close_with_increased_tol(
        var_out, tf_out.make({1, 1, 1}, {50.0}));
    expect_tensor_close_with_increased_tol(
        mean_out, tf_out.make({1, 1, 1}, {11.5}));

    // empty dim list, correction=0 (population), keepdim=true
    optional<ArrayRef<int64_t>> empty_dim_list{ArrayRef<int64_t>{}};
    optional<Scalar> correction_zero(0);
    op_var_mean_correction_out(
        self,
        empty_dim_list,
        correction_zero,
        /*keepdim=*/true,
        var_out,
        mean_out);
    expect_tensor_close_with_increased_tol(
        var_out, tf_out.make({1, 1, 1}, {47.916668}));
    expect_tensor_close_with_increased_tol(
        mean_out, tf_out.make({1, 1, 1}, {11.5}));

    // null dim list, correction=0, keepdim=false
    var_out = tf_out.zeros({});
    mean_out = tf_out.zeros({});
    op_var_mean_correction_out(
        self,
        null_dim_list,
        correction_zero,
        /*keepdim=*/false,
        var_out,
        mean_out);
    expect_tensor_close_with_increased_tol(
        var_out, tf_out.make({}, {47.916668}));
    expect_tensor_close_with_increased_tol(mean_out, tf_out.make({}, {11.5}));

    // empty dim list, correction=1, keepdim=false
    op_var_mean_correction_out(
        self,
        empty_dim_list,
        correction,
        /*keepdim=*/false,
        var_out,
        mean_out);
    expect_tensor_close_with_increased_tol(var_out, tf_out.make({}, {50.0}));
    expect_tensor_close_with_increased_tol(mean_out, tf_out.make({}, {11.5}));
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_invalid_dimensions() {
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
    Tensor var_out = tf_out.zeros({2, 3, 1});
    Tensor mean_out = tf_out.zeros({2, 3, 1});
    optional<Scalar> correction(1);

    // out-of-bound dim
    int64_t dims_1[1] = {3};
    optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_var_mean_correction_out(
            self,
            optional_dim_list,
            correction,
            /*keepdim=*/true,
            var_out,
            mean_out));

    // duplicate dim
    int64_t dims_2[2] = {2, 2};
    optional_dim_list = ArrayRef<int64_t>{dims_2, 2};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_var_mean_correction_out(
            self,
            optional_dim_list,
            correction,
            /*keepdim=*/true,
            var_out,
            mean_out));
  }
};

TEST_F(OpVarMeanCorrectionOutTest, SmokeTest) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpVarMeanCorrectionOutTest, KeepDim) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen supports fewer dtypes";
  }
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_keepdim<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOATHBF16_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarMeanCorrectionOutTest, KeepDim_Aten) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen-specific variant of test case";
  }
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_keepdim<ScalarType::DTYPE, ScalarType::DTYPE>();

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpVarMeanCorrectionOutTest, MultipleDims) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen supports fewer dtypes";
  }
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_multiple_dims<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOATHBF16_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarMeanCorrectionOutTest, MultipleDims_Aten) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen-specific variant of test case";
  }
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_multiple_dims<ScalarType::DTYPE, ScalarType::DTYPE>();

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpVarMeanCorrectionOutTest, NegativeDim) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen supports fewer dtypes";
  }
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_negative_dim<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOATHBF16_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarMeanCorrectionOutTest, NegativeDim_Aten) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen-specific variant of test case";
  }
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_negative_dim<ScalarType::DTYPE, ScalarType::DTYPE>();

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpVarMeanCorrectionOutTest, NullAndEmptyDimList) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen supports fewer dtypes";
  }
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_null_and_empty_dim_list<                                           \
      ScalarType::INPUT_DTYPE,                                            \
      ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOATHBF16_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarMeanCorrectionOutTest, NullAndEmptyDimList_Aten) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen-specific variant of test case";
  }
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_null_and_empty_dim_list<ScalarType::DTYPE, ScalarType::DTYPE>();

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpVarMeanCorrectionOutTest, InvalidDimensionListDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_invalid_dimensions<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_FLOAT_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpVarMeanCorrectionOutTest, InvalidDTypeDies) {
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

  Tensor var_out = tf_float.zeros({2, 3, 1});
  Tensor mean_out = tf_float.zeros({2, 3, 1});
  int64_t dims_1[1] = {2};
  optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
  optional<Scalar> correction(1);

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_var_mean_correction_out(
          self,
          optional_dim_list,
          correction,
          /*keepdim=*/true,
          var_out,
          mean_out));
}

TEST_F(OpVarMeanCorrectionOutTest, EmptyInput) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({2, 0, 3}, {});
  optional<Scalar> correction(1);
  optional<Scalar> correction_zero(0);

  // empty dim list, correction=1, keepdim=true
  optional<ArrayRef<int64_t>> dim_list = ArrayRef<int64_t>{};
  Tensor var_out = tf.zeros({1, 1, 1});
  Tensor mean_out = tf.zeros({1, 1, 1});
  op_var_mean_correction_out(
      x, dim_list, correction, /*keepdim=*/true, var_out, mean_out);
  EXPECT_TENSOR_CLOSE(var_out, tf.make({1, 1, 1}, {NAN}));
  EXPECT_TENSOR_CLOSE(mean_out, tf.make({1, 1, 1}, {NAN}));

  // empty dim list, correction=1, keepdim=false
  var_out = tf.zeros({});
  mean_out = tf.zeros({});
  op_var_mean_correction_out(
      x, dim_list, correction, /*keepdim=*/false, var_out, mean_out);
  EXPECT_TENSOR_CLOSE(var_out, tf.make({}, {NAN}));
  EXPECT_TENSOR_CLOSE(mean_out, tf.make({}, {NAN}));

  // reduce along the empty dim
  int64_t dims1[1] = {1};
  dim_list = ArrayRef<int64_t>{dims1, 1};
  var_out = tf.zeros({2, 3});
  mean_out = tf.zeros({2, 3});
  op_var_mean_correction_out(
      x, dim_list, correction, /*keepdim=*/false, var_out, mean_out);
  EXPECT_TENSOR_CLOSE(var_out, tf.make({2, 3}, {NAN, NAN, NAN, NAN, NAN, NAN}));
  EXPECT_TENSOR_CLOSE(
      mean_out, tf.make({2, 3}, {NAN, NAN, NAN, NAN, NAN, NAN}));

  // reduce along a non-empty dim of an empty tensor
  int64_t dims2[1] = {2};
  dim_list = ArrayRef<int64_t>{dims2, 1};
  var_out = tf.make({2, 0, 1}, {});
  mean_out = tf.make({2, 0, 1}, {});
  op_var_mean_correction_out(
      x, dim_list, correction, /*keepdim=*/true, var_out, mean_out);
  EXPECT_TENSOR_CLOSE(var_out, tf.make({2, 0, 1}, {}));
  EXPECT_TENSOR_CLOSE(mean_out, tf.make({2, 0, 1}, {}));
}

TEST_F(OpVarMeanCorrectionOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({3, 2}, {0.49, 0.40, 0.56, 0.38, 0.49, 0.56});
  Tensor expected_var = tf.make({3}, {0.004050, 0.016200, 0.002450});
  Tensor expected_mean = tf.make({3}, {0.445, 0.47, 0.525});
  optional<Scalar> correction(1);

  Tensor var_out =
      tf.zeros({3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor mean_out =
      tf.zeros({3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_var_mean_correction_out(
      x,
      ArrayRef<int64_t>{1},
      correction,
      /*keepdim=*/false,
      var_out,
      mean_out);
  EXPECT_TENSOR_CLOSE(var_out, expected_var);
  EXPECT_TENSOR_CLOSE(mean_out, expected_mean);
}

TEST_F(OpVarMeanCorrectionOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({3, 2}, {0.49, 0.40, 0.56, 0.38, 0.49, 0.56});
  Tensor expected_var = tf.make({3}, {0.004050, 0.016200, 0.002450});
  Tensor expected_mean = tf.make({3}, {0.445, 0.47, 0.525});
  optional<Scalar> correction(1);

  Tensor var_out =
      tf.zeros({10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor mean_out =
      tf.zeros({10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_var_mean_correction_out(
      x,
      ArrayRef<int64_t>{1},
      correction,
      /*keepdim=*/false,
      var_out,
      mean_out);
  EXPECT_TENSOR_CLOSE(var_out, expected_var);
  EXPECT_TENSOR_CLOSE(mean_out, expected_mean);
}

TEST_F(OpVarMeanCorrectionOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({3, 2}, {0.49, 0.40, 0.56, 0.38, 0.49, 0.56});
  Tensor expected_var = tf.make({3}, {0.004050, 0.016200, 0.002450});
  Tensor expected_mean = tf.make({3}, {0.445, 0.47, 0.525});
  optional<Scalar> correction(1);

  Tensor var_out =
      tf.zeros({1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor mean_out =
      tf.zeros({1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_var_mean_correction_out(
      x,
      ArrayRef<int64_t>{1},
      correction,
      /*keepdim=*/false,
      var_out,
      mean_out);
  EXPECT_TENSOR_CLOSE(var_out, expected_var);
  EXPECT_TENSOR_CLOSE(mean_out, expected_mean);
}

TEST_F(OpVarMeanCorrectionOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Float> tf;
  // clang-format off
  Tensor self = tf.make(
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

  Tensor var_out = tf.zeros({2, 3, 1});
  Tensor mean_out = tf.zeros({2, 3, 1});
  int64_t dims[1] = {-1};
  optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims, 1}};
  optional<Scalar> correction(1);
  op_var_mean_correction_out(
      self,
      optional_dim_list,
      correction,
      /*keepdim=*/true,
      var_out,
      mean_out);
  // All rows contain INFINITY or NAN, so var should be NAN for all rows.
  // Mean can be INFINITY or NAN depending on input values, so only check var.
  // clang-format off
  EXPECT_TENSOR_CLOSE(var_out, tf.make(
    {2, 3, 1},
    {
      NAN,
      NAN,
      NAN,

      NAN,
      NAN,
      NAN,
    }));
  // clang-format on
}
