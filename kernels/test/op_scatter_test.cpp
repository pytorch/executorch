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

class OpScatterSrcOutTest : public OperatorTest {
 protected:
  Tensor& op_scatter_src_out(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      Tensor& out) {
    return torch::executor::aten::scatter_outf(
        context_, self, dim, index, src, out);
  }

  // Common testing for the operator
  template <ScalarType DATA_DTYPE>
  void test_scatter_src_out() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;
    const std::vector<int32_t> sizes = {3, 5};
    // clang-format off
    Tensor src = tf_data.make(
      /*sizes=*/{2, 5},
      {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
      });
    // clang-format on
    Tensor in = tf_data.zeros(sizes);
    Tensor out = tf_data.zeros(sizes);
    // clang-format off
    Tensor index = tf_index.make(
      /*sizes=*/{2, 3},
      {
        0, 1, 2,
        0, 1, 2
      });
    // clang-format on

    // Valid input should give the expected output
    op_scatter_src_out(in, 0, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(
          sizes,
          {
            6, 0, 0, 0, 0,
            0, 7, 0, 0, 0,
            0, 0, 8, 0, 0
          }));
    // clang-format on

    // Valid input should give the expected output
    op_scatter_src_out(in, 1, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(sizes,
        {
          1, 2, 3, 0, 0,
          6, 7, 8, 0, 0,
          0, 0, 0, 0, 0
        }));

    src = tf_data.make(
        /*sizes=*/{2, 3, 3},
        {
          // [0, :, :]
          1,  2,  3,
          4,  5,  6,
          7,  8,  9,

          // [1, :, :]
          10, 11, 12,
          13, 14, 15,
          16, 17, 18
        });
    // clang-format on
    in = tf_data.ones(/*sizes=*/{2, 3, 3});
    out = tf_data.zeros(/*sizes=*/{2, 3, 3});
    // clang-format off
    index = tf_index.make(
      /*sizes=*/{1, 3, 2},
      {
        0, 1,
        1, 2,
        0, 2
      });
    // clang-format on

    op_scatter_src_out(in, 1, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{2, 3, 3},
            {
              // [0, :, :]
              7, 1,  1,
              4, 2,  1,
              1, 8, 1,

              // [1, :, :]
              1, 1,  1,
              1, 1,  1,
              1, 1,  1
            }));
    // clang-format on

    out = tf_data.zeros(/*sizes=*/{2, 3, 3});
    op_scatter_src_out(in, 2, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{2, 3, 3},
            {
              // [0, :, :]
              1, 2, 1,
              1, 4, 5,
              7, 1, 8,

              // [1, :, :]
              1, 1, 1,
              1, 1, 1,
              1, 1, 1
            }));
    // clang-format on
  }

  // Invalid dimensions
  template <ScalarType DATA_DTYPE>
  void test_scatter_src_out_invalid_dim() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;
    const std::vector<int32_t> sizes = {3, 5};
    // clang-format off
    Tensor src = tf_data.make(/*sizes=*/{2, 5},
      {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
      });
    Tensor index = tf_index.make(/*sizes=*/{2, 3},
      {
        0, 1, 2,
        0, 1, 2
      });
    // clang-format on
    Tensor self = tf_data.zeros(sizes);
    Tensor out = tf_data.zeros(sizes);

    // Invalid dim should die
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_src_out(self, -3, index, src, out));
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_src_out(self, 2, index, src, out));

    // Self, index and src hsould have same number of dimensions
    src = tf_data.zeros(/*sizes=*/{2, 2, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_src_out(self, 0, index, src, out));

    src = tf_data.zeros(/*sizes=*/{5, 5});
    index = tf_index.zeros(/*sizes=*/{2, 2, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_src_out(self, 0, index, src, out));

    // Size of dimension of index should be smaller than the size of that
    // dimension of src
    index = tf_index.zeros(/*sizes=*/{4, 6});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_src_out(self, 0, index, src, out));

    // Size of dimension of index should be smaller than the size of that
    // dimension of self if dimension != dim
    index = tf_index.zeros(/*sizes=*/{4, 5});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_src_out(self, 1, index, src, out));

    // Index out of bound for self in dim
    index = tf_index.make(/*sizes=*/{2, 3}, {0, 1, 3, 0, 1, 3});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_src_out(self, 0, index, src, out));
  }
};

class OpScatterValueOutTest : public OperatorTest {
 protected:
  Tensor& op_scatter_value_out(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Scalar& value,
      Tensor& out) {
    return torch::executor::aten::scatter_outf(
        context_, self, dim, index, value, out);
  }

  // Common testing for the operator
  template <ScalarType DATA_DTYPE>
  void test_scatter_value_out() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;

    const Scalar& value = 1;

    const std::vector<int32_t> sizes = {3, 5};
    Tensor self = tf_data.zeros(sizes);
    Tensor out = tf_data.zeros(sizes);
    Tensor index = tf_index.make({2, 3}, {0, 1, 2, 0, 1, 2});

    op_scatter_value_out(self, 0, index, value, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(
          sizes,
          {
            1, 0, 0,  0, 0,
            0, 1, 0,  0, 0,
            0, 0, 1, 0, 0
          }));
    // clang-format on

    op_scatter_value_out(self, 1, index, value, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(sizes,
        {
          1, 1, 1, 0, 0,
          1, 1, 1, 0, 0,
          0, 0, 0, 0, 0
        }));

    const Scalar& value2 = 2;
    self = tf_data.ones(/*sizes=*/{2, 3, 3});
    out = tf_data.zeros(/*sizes=*/{2, 3, 3});
    // clang-format off
    index = tf_index.make(
      /*sizes=*/{1, 3, 2},
      {
        0, 1,
        1, 2,
        0, 2
      });
    // clang-format on

    op_scatter_value_out(self, 1, index, value2, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{2, 3, 3},
            {
              // [0, :, :]
              2, 1, 1,
              2, 2, 1,
              1, 2, 1,

              // [1, :, :]
              1, 1, 1,
              1, 1, 1,
              1, 1, 1
            }));
    // clang-format on

    out = tf_data.zeros(/*sizes=*/{2, 3, 3});
    op_scatter_value_out(self, 2, index, value2, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{2, 3, 3},
            {
              // [0, :, :]
              2, 2, 1,
              1, 2, 2,
              2, 1, 2,

              // [1, :, :]
              1, 1, 1,
              1, 1, 1,
              1, 1, 1
            }));
    // clang-format on
  }

  // Invalid dimensions
  template <ScalarType DATA_DTYPE>
  void test_scatter_value_out_invalid_dim() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;
    // clang-format off
    Tensor self = tf_data.make(/*sizes=*/{2, 5},
      {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
      });
    const std::vector<int32_t> sizes = {2, 3};
    Tensor index = tf_index.make(sizes,
      {
        0, 1, 0,
        1, 0, 1,
      });
    // clang-format on
    const Scalar& value = 1;
    Tensor out = tf_data.zeros(sizes);

    // Invalid dim should die
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_value_out(self, -3, index, value, out));
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_value_out(self, 2, index, value, out));

    // Self and index hsould have same number of dimensions
    index = tf_index.zeros(/*sizes=*/{2, 2, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_value_out(self, 0, index, value, out));

    // Size of dimension of index should be smaller than the size of that
    // dimension of self if dimension != dim
    index = tf_index.zeros(/*sizes=*/{3, 5});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_value_out(self, 1, index, value, out));

    // Index out of bound for self in dim
    index = tf_index.make(/*sizes=*/{2, 3}, {0, 1, 2, 0, 1, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_value_out(self, 0, index, value, out));
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    TensorFactory<ScalarType::Int> tf;
    TensorFactory<ScalarType::Long> tf_index;

    Tensor input = tf.ones({2, 3, 4});
    Tensor index = tf_index.zeros({2, 3, 4});
    const Scalar& value = 1;
    Tensor expected = tf.ones({2, 3, 4});
    Tensor out = tf.zeros(out_shape, dynamism);

    op_scatter_value_out(input, 2, index, value, out);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpScatterSrcOutTest, AllValidInputOutputSupport) {
#define TEST_ENTRY(CTYPE, DTYPE) test_scatter_src_out<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpScatterSrcOutTest, InvalidDimensionsDies) {
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_scatter_src_out_invalid_dim<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpScatterValueOutTest, AllValidInputOutputSupport) {
#define TEST_ENTRY(CTYPE, DTYPE) test_scatter_value_out<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpScatterValueOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;
  // clang-format off
  Tensor self = tf_data.make(
      /*sizes=*/{2, 5},
      {
        0.0, -INFINITY,        NAN,      2.33, NAN,
        NAN,  INFINITY,  -INFINITY, -INFINITY, 2.33
      });
  // clang-format on
  Tensor index = tf_index.make({2, 3}, {0, 1, 0, 1, 0, 1});
  const Scalar& value = INFINITY;
  Tensor out = tf_data.zeros({2, 5});

  // Valid input should give the expected output
  op_scatter_value_out(self, 0, index, value, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(
      out,
      tf_data.make(/*sizes=*/{2, 5},
      {
        INFINITY, INFINITY, INFINITY,      2.33, NAN,
        INFINITY, INFINITY, INFINITY, -INFINITY, 2.33
      }));
  // clang-format on
}

TEST_F(OpScatterValueOutTest, InvalidDimensionsDies) {
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_scatter_value_out_invalid_dim<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpScatterValueOutTest, MismatchedInputDtypesDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf_char.make({2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  const std::vector<int32_t> sizes = {2, 3};
  Tensor index = tf_byte.make(sizes, {0, 1, 0, 0, 1, 0});
  const Scalar& value = 5;
  Tensor out = tf_char.zeros(sizes);

  // Types other than long for index should die
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_value_out(self, 0, index, value, out));

  // Mismatched dtype of self and out should die
  self = tf_byte.make(/*sizes=*/{2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  index = tf_long.make(sizes, {0, 1, 0, 1, 0, 1});
  out = tf_char.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_value_out(self, 0, index, value, out));
}

TEST_F(OpScatterValueOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpScatterValueOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpScatterValueOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}

TEST_F(OpScatterValueOutTest, EmptyIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.ones({2, 5});
  Tensor index = tf_index.zeros({2, 0, 3});
  const Scalar& value = 5;
  Tensor out = tf_data.zeros({2, 5});
  op_scatter_value_out(self, 0, index, value, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.ones({2, 5}));
}

TEST_F(OpScatterValueOutTest, ValidZeroDim) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  Tensor index = tf_index.zeros({});
  const Scalar& value = 5;
  Tensor out = tf_data.zeros({});
  op_scatter_value_out(self, 0, index, value, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({}, {5}));
}

TEST_F(OpScatterValueOutTest, InvalidZeroDimInput) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.ones({});
  Tensor index = tf_index.make({2, 3}, {0, 0, 0, 0, 0, 0});
  const Scalar& value = 5;
  Tensor out = tf_data.zeros({});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_value_out(self, 0, index, value, out));
}

TEST_F(OpScatterValueOutTest, InvalidZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor index = tf_index.make({}, {2});
  const Scalar& value = 5;
  Tensor out = tf_data.zeros({2, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_value_out(self, 1, index, value, out));
}

TEST_F(OpScatterValueOutTest, ValidZeroDimInputAndOneDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  Tensor index = tf_index.make({3}, {0, 0, 0});
  const Scalar& value = 5;
  Tensor out = tf_data.make({}, {2.71});
  op_scatter_value_out(self, 0, index, value, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({}, {5}));
}

TEST_F(OpScatterValueOutTest, ValidOneDimInputAndZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({3}, {10, 20, 30});
  Tensor index = tf_index.make({}, {2});
  const Scalar& value = 5;
  Tensor out = tf_data.make({3}, {1729, 1729, 1729});
  op_scatter_value_out(self, 0, index, value, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({3}, {10, 20, 5}));
}

TEST_F(OpScatterValueOutTest, InvalidZeroDimInputAndOneDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  Tensor index = tf_index.make({3}, {10, 100, 1000});
  const Scalar& value = 5;
  Tensor out = tf_data.make({}, {2.71});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_value_out(self, 0, index, value, out));
}

TEST_F(OpScatterValueOutTest, InvalidOneDimInputAndZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({3}, {10, 20, 30});
  Tensor index = tf_index.make({}, {100});
  const Scalar& value = 5;
  Tensor out = tf_data.make({3}, {1729, 1729, 1729});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_value_out(self, 0, index, value, out));
}

TEST_F(OpScatterSrcOutTest, EmptyIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.ones({2, 5});
  Tensor index = tf_index.zeros({2, 0, 3});
  Tensor src = tf_data.ones({1, 1, 4});
  Tensor out = tf_data.zeros({2, 5});
  op_scatter_src_out(self, 0, index, src, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.ones({2, 5}));
}

TEST_F(OpScatterSrcOutTest, ValidZeroDim) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  Tensor index = tf_index.zeros({});
  Tensor src = tf_data.make({}, {5});
  Tensor out = tf_data.zeros({});
  op_scatter_src_out(self, 0, index, src, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({}, {5}));
}

TEST_F(OpScatterSrcOutTest, InvalidZeroDimInput) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.ones({});
  Tensor index = tf_index.make({2, 3}, {0, 0, 0, 0, 0, 0});
  Tensor src = tf_data.make({}, {5});
  Tensor out = tf_data.zeros({});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_src_out(self, 0, index, src, out));
}

TEST_F(OpScatterSrcOutTest, InvalidZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor index = tf_index.make({}, {2});
  Tensor src = tf_data.make({}, {5});
  Tensor out = tf_data.zeros({2, 3});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_src_out(self, 1, index, src, out));
}

TEST_F(OpScatterSrcOutTest, ValidZeroDimInputAndOneDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  Tensor index = tf_index.make({3}, {0, 0, 0});
  Tensor src = tf_data.make({3}, {5, 5, 5});
  Tensor out = tf_data.make({}, {2.71});
  op_scatter_src_out(self, 0, index, src, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({}, {5}));
}

TEST_F(OpScatterSrcOutTest, ValidOneDimInputAndZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({3}, {10, 20, 30});
  Tensor index = tf_index.make({}, {2});
  Tensor src = tf_data.make({}, {5});
  Tensor out = tf_data.make({3}, {1729, 1729, 1729});
  op_scatter_src_out(self, 0, index, src, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({3}, {10, 20, 5}));
}

TEST_F(OpScatterSrcOutTest, InvalidZeroDimInputAndOneDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  Tensor index = tf_index.make({3}, {10, 100, 1000});
  Tensor src = tf_data.make({}, {5});
  Tensor out = tf_data.make({}, {2.71});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_src_out(self, 0, index, src, out));
}

TEST_F(OpScatterSrcOutTest, InvalidOneDimInputAndZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({3}, {10, 20, 30});
  Tensor index = tf_index.make({}, {100});
  Tensor src = tf_data.make({}, {5});
  Tensor out = tf_data.make({3}, {1729, 1729, 1729});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_src_out(self, 0, index, src, out));
}
