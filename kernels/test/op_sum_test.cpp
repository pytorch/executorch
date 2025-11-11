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
using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpSumOutTest : public OperatorTest {
 protected:
  Tensor& op_sum_intlist_out(
      const Tensor& self,
      optional<ArrayRef<int64_t>> dim,
      bool keepdim,
      optional<ScalarType> dtype,
      Tensor& out) {
    return torch::executor::aten::sum_outf(
        context_, self, dim, keepdim, dtype, out);
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_sum_dim_out_invalid_dimensions() {
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
        op_sum_intlist_out(
            self, optional_dim_list, /*keepdim=*/true, dtype, out));

    // the same dim appears multiple times in list of dims
    int64_t dims_2[2] = {2, 2};
    optional_dim_list = ArrayRef<int64_t>{dims_2, 2};
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_sum_intlist_out(
            self, optional_dim_list, /*keepdim=*/true, dtype, out));
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_sum_dim_out_invalid_shape() {
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
        op_sum_intlist_out(
            self, optional_dim_list, /*keepdim=*/true, dtype, out));

    // dimension size mismatch when keepdim is false
    out = tf_out.zeros({2, 1, 4});
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_sum_intlist_out(
            self, optional_dim_list, /*keepdim=*/false, dtype, out));
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_complex_dtype() {
    TensorFactory<DTYPE> tf;

    Tensor self = tf.make(
        {2, 3, 2},
        {CTYPE(1, 1),
         CTYPE(2, 2),
         CTYPE(3, 3),
         CTYPE(4, 4),
         CTYPE(5, 5),
         CTYPE(6, 6),

         CTYPE(7, 7),
         CTYPE(8, 8),
         CTYPE(9, 9),
         CTYPE(10, 10),
         CTYPE(11, 11),
         CTYPE(12, 12)});

    Tensor out1 = tf.make(
        {2, 3, 1},
        {
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
        });
    int64_t dims_1[1] = {2};
    optional<ArrayRef<int64_t>> dim_list1{ArrayRef<int64_t>{dims_1, 1}};
    optional<ScalarType> dtype = DTYPE;

    op_sum_intlist_out(self, dim_list1, true, dtype, out1);

    Tensor expected1 = tf.make(
        {2, 3, 1},
        {CTYPE(3, 3),
         CTYPE(7, 7),
         CTYPE(11, 11),

         CTYPE(15, 15),
         CTYPE(19, 19),
         CTYPE(23, 23)});

    EXPECT_TENSOR_CLOSE(out1, expected1);

    Tensor out2 = tf.make(
        {2, 1, 2},
        {
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
            CTYPE(0, 0),
        });
    int64_t dims_2[1] = {1};
    optional<ArrayRef<int64_t>> dim_list2{ArrayRef<int64_t>{dims_2, 1}};

    op_sum_intlist_out(self, dim_list2, true, dtype, out2);

    Tensor expected2 = tf.make(
        {2, 1, 2}, {CTYPE(9, 9), CTYPE(12, 12), CTYPE(27, 27), CTYPE(30, 30)});
    EXPECT_TENSOR_CLOSE(out2, expected2);

    Tensor out3 = tf.make(
        {1, 1, 1},
        {
            CTYPE(0, 0),
        });
    optional<ArrayRef<int64_t>> null_dim_list;

    op_sum_intlist_out(self, null_dim_list, true, dtype, out3);

    Tensor expected3 = tf.make({1, 1, 1}, {CTYPE(78, 78)});

    EXPECT_TENSOR_CLOSE(out3, expected3);
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_sum_dim_out_dtype() {
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
    op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 3, 1},
      {
        6,
        22,
        38,

        54,
        70,
        86
      }));
    // clang-format on

    // keepdim=false should work
    out = tf_out.zeros({2, 3});
    op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/false, dtype, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 3},
      {
        6,  22, 38,
        54, 70, 86
      }));
    // clang-format on

    // dim list with multiple dimensions should work
    out = tf_out.zeros({1, 1, 4});
    int64_t dims_01[2] = {0, 1};
    optional_dim_list = ArrayRef<int64_t>{dims_01, 2};
    op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 4}, {60, 66, 72, 78}));

    out = tf_out.zeros({4});
    op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/false, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({4}, {60, 66, 72, 78}));

    out = tf_out.zeros({1, 3, 1});
    int64_t dims_02[2] = {0, 2};
    optional_dim_list = ArrayRef<int64_t>{dims_02, 2};
    op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 3, 1}, {60, 92, 124}));

    out = tf_out.zeros({3});
    op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/false, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({3}, {60, 92, 124}));

    // dim list with negative dimensions should work
    out = tf_out.zeros({2, 1, 4});
    int64_t dims_3[1] = {-2};
    optional_dim_list = ArrayRef<int64_t>{dims_3, 1};
    op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 1, 4},
      {
        12, 15, 18, 21,

        48, 51, 54, 57,
      }));
    // clang-format on

    // empty/null dim list should work
    // clang-format off
    self = tf_in.make(
      {2, 2, 4},
      {
        0, 1, 2, 3,
        4, 5, 6, 7,

        0, 1, 2, 3,
        4, 5, 6, 7,
      });
    // clang-format on
    out = tf_out.zeros({1, 1, 1});
    optional<ArrayRef<int64_t>> null_dim_list;
    op_sum_intlist_out(self, null_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 1}, {56}));

    optional<ArrayRef<int64_t>> empty_dim_list{ArrayRef<int64_t>{}};
    op_sum_intlist_out(self, empty_dim_list, /*keepdim=*/true, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({1, 1, 1}, {56}));

    out = tf_out.zeros({});
    op_sum_intlist_out(self, null_dim_list, /*keepdim=*/false, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({}, {56}));

    op_sum_intlist_out(self, empty_dim_list, /*keepdim=*/false, dtype, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make({}, {56}));
  }
};

TEST_F(OpSumOutTest, InvalidDimensionListDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_sum_dim_out_invalid_dimensions<                                    \
      ScalarType::INPUT_DTYPE,                                            \
      ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_REAL_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpSumOutTest, InvalidShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_sum_dim_out_invalid_shape<                                         \
      ScalarType::INPUT_DTYPE,                                            \
      ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_REAL_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpSumOutTest, MismatchedDTypesDies) {
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

  Tensor out = tf_float.zeros({2, 3, 1});
  int64_t dims_1[1] = {2};
  optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
  optional<ScalarType> dtype = ScalarType::Double;

  // out tensor should be of the same dtype with dtype when dtype is specified
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_sum_intlist_out(
          self, optional_dim_list, /*keepdim=*/true, dtype, out));
}

TEST_F(OpSumOutTest, AllRealInputRealOutputPasses) {
  // Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_sum_dim_out_dtype<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_REALHBF16_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpSumOutTest, TypeConversionTest) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Int> tf_int;
  // clang-format off
  Tensor self = tf_int.make(
    {2, 3, 4},
    {
      0, 0, 0, 0,
      2, 2, 2, 2,
      4, 4, 4, 4,

      8,  8,  8,  8,
      16, 16, 16, 16,
      64, 64, 64, 64,
    });
  // clang-format on

  int64_t dims_1[1] = {2};
  optional<ArrayRef<int64_t>> optional_dim_list{ArrayRef<int64_t>{dims_1, 1}};
  optional<ScalarType> dtype;

  // int -> bool conversion should work
  Tensor out = tf_bool.zeros({2, 3, 1});
  op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(out, tf_bool.make(
    {2, 3, 1},
    {
      false,
      true,
      true,

      true,
      true,
      true
    }));
  // clang-format on

  // int -> byte conversion should work
  out = tf_byte.zeros({2, 3, 1});
  op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(out, tf_byte.make(
    {2, 3, 1},
    {
      0,
      8,
      16,

      32,
      64,
      0,
    }));
  // clang-format on
}

TEST_F(OpSumOutTest, AllComplexDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_complex_dtype<ctype, ScalarType::dtype>();
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    ET_FORALL_COMPLEX_TYPES(TEST_ENTRY);
  } else {
    ET_FORALL_COMPLEXH_TYPES(TEST_ENTRY);
  }
#undef TEST_ENTRY
}

TEST_F(OpSumOutTest, InfinityAndNANTest) {
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
  op_sum_intlist_out(self, optional_dim_list, /*keepdim=*/true, dtype, out);
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

TEST_F(OpSumOutTest, EmptyInput) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({2, 0, 3}, {});
  optional<ScalarType> dtype = ScalarType::Float;
  optional<ArrayRef<int64_t>> dim_list = ArrayRef<int64_t>{};
  Tensor out = tf.ones({1, 1, 1});
  op_sum_intlist_out(x, dim_list, /*keepdim=*/true, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.zeros({1, 1, 1}));

  out = tf.ones({});
  op_sum_intlist_out(x, dim_list, /*keepdim=*/false, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.zeros({}));

  int64_t dims1[1] = {1};
  dim_list = ArrayRef<int64_t>{dims1, 1};
  out = tf.ones({2, 3});
  op_sum_intlist_out(x, dim_list, /*keepdim=*/false, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.zeros({2, 3}));

  int64_t dims2[1] = {2};
  dim_list = ArrayRef<int64_t>{dims2, 1};
  out = tf.make({2, 0, 1}, {});
  op_sum_intlist_out(x, dim_list, /*keepdim=*/true, dtype, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 0, 1}, {}));
}
