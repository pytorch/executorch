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
#include <executorch/kernels/test/supported_features_skip.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
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

  Tensor& op_mean_dtype_out(
      const Tensor& self,
      optional<ScalarType> dtype,
      Tensor& out) {
    return torch::executor::aten::mean_outf(context_, self, dtype, out);
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
    test_mean_dim_out_dtype<ScalarType::Bool, ScalarType::Half>() {
  test_mean_dim_out_bool<ScalarType::Half>();
}

template <>
void OpMeanOutTest::
    test_mean_dim_out_dtype<ScalarType::Bool, ScalarType::BFloat16>() {
  test_mean_dim_out_bool<ScalarType::BFloat16>();
}

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

TEST_F(OpMeanOutTest, BFloat16GenericPathAccumulatesInFloat) {
  TensorFactory<ScalarType::BFloat16> tf;
  // Reducing dim=0 of {512, 1} is not the last dim, so the generic path is
  // taken. Without fp32 accumulation the sum saturates at ~256, giving
  // 256/512 = 0.5 instead of 1.0.
  constexpr int N = 512;
  Tensor x = tf.ones({N, 1});
  Tensor out = tf.zeros({1});
  int64_t dim = 0;
  op_mean_out(
      x, ArrayRef<int64_t>{&dim, 1}, /*keepdim=*/false, /*dtype=*/{}, out);
  Tensor expected = tf.full({1}, 1.0f);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpMeanOutTest, BFloat16LargeDimAccumulatesInFloat) {
  TensorFactory<ScalarType::BFloat16> tf;
  // N=512, all-ones input: without fp32 accumulation the sum saturates at
  // ~256 in BFloat16, giving 256/512 = 0.5 instead of 1.0.
  constexpr int N = 512;
  Tensor x = tf.ones({1, N});
  Tensor out = tf.zeros({1});
  int64_t dim = 1;
  op_mean_out(
      x, ArrayRef<int64_t>{&dim, 1}, /*keepdim=*/false, /*dtype=*/{}, out);
  Tensor expected = tf.full({1}, 1.0f);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpMeanOutTest, ChannelsLastSpatialReduction) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data(120);
  std::iota(data.begin(), data.end(), 0.0f);
  Tensor x = tf.channels_last_like(tf.make({2, 3, 4, 5}, data));
  Tensor out = tf.zeros({2, 3});
  const int64_t dims[] = {2, 3};

  op_mean_out(x, ArrayRef<int64_t>(dims), false, {}, out);

  EXPECT_TENSOR_CLOSE(
      out, tf.make({2, 3}, {9.5, 29.5, 49.5, 69.5, 89.5, 109.5}));
}

TEST_F(OpMeanOutTest, ChannelsLastReductionDimensions) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data(120);
  std::iota(data.begin(), data.end(), 0.0f);
  Tensor contiguous = tf.make({2, 3, 4, 5}, data);
  Tensor channels_last = tf.channels_last_like(contiguous);

  for (int mask = 0; mask < 16; ++mask) {
    SCOPED_TRACE(mask);
    std::vector<int64_t> dims;
    for (int d = 3; d >= 0; --d) {
      if (mask & (1 << d)) {
        dims.push_back(d - 4);
      }
    }
    const ArrayRef<int64_t> dim_list(dims.data(), dims.size());
    for (bool keepdim : {false, true}) {
      SCOPED_TRACE(keepdim);
      std::vector<int32_t> sizes;
      for (int d = 0; d < 4; ++d) {
        if (mask == 0 || (mask & (1 << d))) {
          if (keepdim) {
            sizes.push_back(1);
          }
        } else {
          sizes.push_back(contiguous.size(d));
        }
      }
      Tensor expected = tf.zeros(sizes);
      op_mean_out(contiguous, dim_list, keepdim, {}, expected);

      Tensor out = tf.zeros(sizes);
      op_mean_out(channels_last, dim_list, keepdim, {}, out);
      EXPECT_TENSOR_CLOSE(out, expected);

      if (keepdim) {
        Tensor expected_channels_last = tf.channels_last_like(expected);
        for (const Tensor& input : {contiguous, channels_last}) {
          Tensor out_channels_last = tf.zeros_channels_last(sizes);
          op_mean_out(input, dim_list, keepdim, {}, out_channels_last);
          EXPECT_TENSOR_CLOSE(out_channels_last, expected_channels_last);
        }
      }
    }
  }
}

TEST_F(OpMeanOutTest, ChannelsLastAllDimensions) {
  TensorFactory<ScalarType::Float> tf;
  Tensor x =
      tf.channels_last_like(tf.make({1, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}));
  Tensor out = tf.zeros({});
  op_mean_out(x, {}, false, {}, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {3.5}));

  op_mean_dtype_out(x, {}, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {3.5}));

  Tensor out_keepdim = tf.zeros_channels_last({1, 1, 1, 1});
  op_mean_out(x, {}, true, {}, out_keepdim);
  EXPECT_TENSOR_CLOSE(out_keepdim, tf.make_channels_last({1, 1, 1, 1}, {3.5}));
}

TEST_F(OpMeanOutTest, ChannelsLast3d) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<float> data(48);
  std::iota(data.begin(), data.end(), 0.0f);
  Tensor x = tf.make_with_dimorder({2, 3, 2, 2, 2}, data, {0, 2, 3, 4, 1});
  Tensor out = tf.zeros({2, 3});
  const int64_t dims[] = {2, 3, 4};
  op_mean_out(x, ArrayRef<int64_t>(dims), false, {}, out);
  EXPECT_TENSOR_CLOSE(
      out, tf.make({2, 3}, {10.5, 11.5, 12.5, 34.5, 35.5, 36.5}));

  Tensor out_keepdim = tf.make_with_dimorder(
      {1, 3, 2, 2, 2}, std::vector<float>(24), {0, 2, 3, 4, 1});
  op_mean_out(x, ArrayRef<int64_t>{0}, true, {}, out_keepdim);
  std::vector<float> expected_data(24);
  std::iota(expected_data.begin(), expected_data.end(), 12.0f);
  EXPECT_TENSOR_CLOSE(
      out_keepdim,
      tf.make_with_dimorder({1, 3, 2, 2, 2}, expected_data, {0, 2, 3, 4, 1}));

  Tensor out_4d = tf.zeros_channels_last({2, 3, 2, 2});
  op_mean_out(x, ArrayRef<int64_t>{-1}, false, {}, out_4d);
  EXPECT_TENSOR_CLOSE(
      out_4d,
      tf.make_channels_last(
          {2, 3, 2, 2}, {1.5,  2.5,  3.5,  7.5,  8.5,  9.5,  13.5, 14.5,
                         15.5, 19.5, 20.5, 21.5, 25.5, 26.5, 27.5, 31.5,
                         32.5, 33.5, 37.5, 38.5, 39.5, 43.5, 44.5, 45.5}));
}

TEST_F(OpMeanOutTest, ChannelsLastDTypeConversion) {
  TensorFactory<ScalarType::Int> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  Tensor x = tf_in.channels_last_like(
      tf_in.make({1, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}));
  Tensor out = tf_out.zeros_channels_last({1, 2, 2, 1});
  op_mean_out(x, ArrayRef<int64_t>{3}, true, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.channels_last_like(
          tf_out.make({1, 2, 2, 1}, {0.5, 2.5, 4.5, 6.5})));
}

TEST_F(OpMeanOutTest, ChannelsLastBFloat16AccumulatesInFloat) {
  TensorFactory<ScalarType::BFloat16> tf;
  Tensor x = tf.full_channels_last({1, 2, 16, 32}, 1);
  Tensor out = tf.zeros({1, 2});
  const int64_t dims[] = {2, 3};
  op_mean_out(x, ArrayRef<int64_t>(dims), false, {}, out);
  EXPECT_TENSOR_CLOSE(out, tf.ones({1, 2}));
}

TEST_F(OpMeanOutTest, ChannelsLastEmptyInput) {
  TensorFactory<ScalarType::Float> tf;
  Tensor x = tf.make_channels_last({2, 3, 0, 4}, {});
  Tensor out = tf.zeros_channels_last({2, 3, 1, 4});
  op_mean_out(x, ArrayRef<int64_t>{2}, true, {}, out);
  EXPECT_TENSOR_CLOSE(out, tf.full_channels_last({2, 3, 1, 4}, NAN));

  Tensor empty_out = tf.make_channels_last({2, 3, 0, 1}, {});
  op_mean_out(x, ArrayRef<int64_t>{3}, true, {}, empty_out);
  EXPECT_TENSOR_CLOSE(empty_out, tf.make_channels_last({2, 3, 0, 1}, {}));

  Tensor scalar_out = tf.zeros({});
  op_mean_dtype_out(x, {}, scalar_out);
  EXPECT_TENSOR_CLOSE(scalar_out, tf.make({}, {NAN}));
}

TEST_F(OpMeanOutTest, ChannelsLastDynamicOutput) {
  TensorFactory<ScalarType::Float> tf;
  Tensor x =
      tf.channels_last_like(tf.make({1, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}));
  Tensor out = tf.zeros_channels_last(
      {2, 3, 4, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_mean_out(x, ArrayRef<int64_t>{3}, true, {}, out);
  EXPECT_TENSOR_CLOSE(
      out, tf.channels_last_like(tf.make({1, 2, 2, 1}, {0.5, 2.5, 4.5, 6.5})));
}

TEST_F(OpMeanOutTest, InvalidDimensionListDies) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen kernel test fails");
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
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen kernel test fails");
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
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "ATen kernel test fails");
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
  ET_FORALL_FLOATHBF16_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REALHBBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpMeanOutTest, HalfSupport) {
  ET_SKIP_IF(
      torch::executor::testing::SupportedFeatures::get()->is_aten,
      "Test Half support only for ExecuTorch mode");
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

// DISABLED: Dynamic shape unbound not supported
TEST_F(OpMeanOutTest, DISABLED_DynamicShapeUnbound) {
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

TEST_F(OpMeanOutTest, DTypeOutFloatValid) {
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
  Tensor expected_result = tf.make({}, {1.0});

  Tensor out = tf.zeros({});
  Tensor ret = op_mean_dtype_out(x, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMeanOutTest, DTypeOutFloatToBoolInvalid) {
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
  Tensor expected_result = tf.make({}, {1.0});

  Tensor out = tf.zeros({});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_mean_dtype_out(x, ScalarType::Bool, out));
}

TEST_F(OpMeanOutTest, DTypeOutFloatInfinity) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({2, 1}, {INFINITY, INFINITY});
  Tensor expected_result = tf.make({}, {INFINITY});

  Tensor out = tf.zeros({});

  Tensor ret = op_mean_dtype_out(x, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMeanOutTest, DTypeOutFloatNAN) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({2, 1}, {NAN, INFINITY});
  Tensor expected_result = tf.make({}, {NAN});

  Tensor out = tf.zeros({});

  Tensor ret = op_mean_dtype_out(x, ScalarType::Float, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMeanOutTest, EmptyInput) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({2, 0, 3}, {});
  optional<ScalarType> dtype = ScalarType::Float;
  optional<ArrayRef<int64_t>> dim_list = ArrayRef<int64_t>{};
  Tensor out = tf.zeros({1, 1, 1});
  op_mean_out(x, dim_list, /*keepdim=*/true, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {NAN}));

  out = tf.zeros({});
  op_mean_out(x, dim_list, /*keepdim=*/false, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {NAN}));

  int64_t dims1[1] = {1};
  dim_list = ArrayRef<int64_t>{dims1, 1};
  out = tf.zeros({2, 3});
  op_mean_out(x, dim_list, /*keepdim=*/false, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {NAN, NAN, NAN, NAN, NAN, NAN}));

  int64_t dims2[1] = {2};
  dim_list = ArrayRef<int64_t>{dims2, 1};
  out = tf.make({2, 0, 1}, {});
  op_mean_out(x, dim_list, /*keepdim=*/true, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 0, 1}, {}));
}
