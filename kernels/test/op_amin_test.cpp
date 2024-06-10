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

class OpAminOutTest : public OperatorTest {
 protected:
  Tensor& op_amin_out(
      const Tensor& in,
      ArrayRef<int64_t> dim,
      bool keepdim,
      Tensor& out) {
    return torch::executor::aten::amin_outf(context_, in, dim, keepdim, out);
  }

  template <ScalarType DTYPE>
  void test_amin_out_invalid_dimensions() {
    TensorFactory<DTYPE> tf;

    // clang-format off
    Tensor in = tf.make(
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
    Tensor out = tf.zeros({2, 3, 1});

    // out-of-bound dim in dim list
    int64_t dims_1[1] = {3};
    ArrayRef<int64_t> dim_list{ArrayRef<int64_t>{dims_1, 1}};
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_amin_out(in, dim_list, /*keepdim=*/true, out));

    // the same dim appears multiple times in list of dims
    int64_t dims_2[2] = {2, 2};
    dim_list = ArrayRef<int64_t>{dims_2, 2};
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_amin_out(in, dim_list, /*keepdim=*/true, out));
  }

  template <ScalarType DTYPE>
  void test_amin_out_invalid_shape() {
    TensorFactory<DTYPE> tf;

    // clang-format off
    Tensor in = tf.make(
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

    // dimension size mismatch when keepdim is true
    Tensor out = tf.zeros({2, 4});

    int64_t dims_1[1] = {1};
    ArrayRef<int64_t> dim_list{ArrayRef<int64_t>{dims_1, 1}};
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_amin_out(in, dim_list, /*keepdim=*/true, out));

    // dimension size mismatch when keepdim is false
    out = tf.zeros({2, 1, 4});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_amin_out(in, dim_list, /*keepdim=*/false, out));
  }

  template <ScalarType DTYPE>
  void test_amin_out_dtype() {
    TensorFactory<DTYPE> tf;
    // clang-format off
    Tensor in = tf.make(
      {2, 3, 4},
      {
        0, 1, 2, 4,
        4, 2, 1, 0,
        1, 5, 4, 2,

        4, 2, 1, 0,
        5, 1, 2, 4,
        7, 5, 4, 2,
      });
    // clang-format on

    // keepdim=true should work
    Tensor out = tf.zeros({2, 3, 1});
    int64_t dims_1[1] = {2};
    ArrayRef<int64_t> dim_list{ArrayRef<int64_t>{dims_1, 1}};

    op_amin_out(in, dim_list, /*keepdim=*/true, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf.make(
      {2, 3, 1},
      {0, 0, 1, 0, 1, 2}));
    // clang-format on

    // keepdim=false should work
    out = tf.zeros({2, 3});
    op_amin_out(in, dim_list, /*keepdim=*/false, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf.make(
      {2, 3},
      {0, 0, 1, 0, 1, 2}));
    // clang-format on

    // dim list with multiple dimensions should work
    out = tf.zeros({1, 1, 4});
    int64_t dims_2[2] = {0, 1};
    dim_list = ArrayRef<int64_t>{dims_2, 2};
    op_amin_out(in, dim_list, /*keepdim=*/true, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 4}, {0, 1, 1, 0}));

    out = tf.zeros({4});
    op_amin_out(in, dim_list, /*keepdim=*/false, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({4}, {0, 1, 1, 0}));

    // dim list with negative dimensions should work
    out = tf.zeros({2, 1, 4});
    int64_t dims_3[1] = {-2};
    dim_list = ArrayRef<int64_t>{dims_3, 1};
    op_amin_out(in, dim_list, /*keepdim=*/true, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf.make(
      {2, 1, 4},
      {
        0, 1, 1, 0,

        4, 1, 1, 0,
      }));
    // clang-format on

    // empty/null dim list should work
    // clang-format off
    in = tf.make(
      {2, 2, 4},
      {
        8, 7, 5, 4,
        4, 3, 7, 9,

        4, 2, 6, 8,
        8, 7, 3, 4,
      });
    // clang-format on
    out = tf.zeros({1, 1, 1});
    ArrayRef<int64_t> null_dim_list;
    op_amin_out(in, null_dim_list, /*keepdim=*/true, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {2}));

    ArrayRef<int64_t> empty_dim_list{ArrayRef<int64_t>{}};
    op_amin_out(in, empty_dim_list, /*keepdim=*/true, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({1, 1, 1}, {2}));

    out = tf.zeros({});
    op_amin_out(in, null_dim_list, /*keepdim=*/false, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({}, {2}));

    op_amin_out(in, empty_dim_list, /*keepdim=*/false, out);
    EXPECT_TENSOR_CLOSE(out, tf.make({}, {2}));
  }
};

template <>
void OpAminOutTest::test_amin_out_dtype<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf_bool;
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

  Tensor out = tf_bool.zeros({2, 3, 1});

  // +/-inf and nan should work
  op_amin_out(in, /*dim=*/-1, /*keepdim=*/true, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(
      out, tf_bool.make(
        {2, 3, 1},
        {
          false,
          false,
          false,

          false,
          false,
          true
        }));
  // clang-format on
}

TEST_F(OpAminOutTest, InvalidDimensionListDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_amin_out_invalid_dimensions<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAminOutTest, InvalidShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_amin_out_invalid_shape<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAminOutTest, MismatchedDTypesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Int> tf_int;

  // clang-format off
  Tensor in = tf_int.make(
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

  Tensor out = tf_float.zeros({2, 3, 1});
  int64_t dims_1[1] = {2};
  ArrayRef<int64_t> dim_list{ArrayRef<int64_t>{dims_1, 1}};

  // out tensor should be of the same dtype with dtype when dtype is specified
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_amin_out(in, dim_list, /*keepdim=*/true, out));
}

TEST_F(OpAminOutTest, AllRealInputOutputPasses) {
#define TEST_ENTRY(ctype, dtype) test_amin_out_dtype<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAminOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Float> tf_float;
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

  Tensor out = tf_float.zeros({2, 3, 1});
  int64_t dims[1] = {-1};
  ArrayRef<int64_t> dim_list{ArrayRef<int64_t>{dims, 1}};
  op_amin_out(in, dim_list, /*keepdim=*/true, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(
      out, tf_float.make({2, 3, 1}, {0, -INFINITY, NAN, NAN, NAN, NAN}));
  // clang-format on
}
