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

class OpGatherOutTest : public OperatorTest {
 protected:
  Tensor& op_gather_out(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      bool sparse_grad,
      Tensor& out) {
    return torch::executor::aten::gather_outf(
        context_, self, dim, index, sparse_grad, out);
  }

  // Common testing for the operator
  template <ScalarType DATA_DTYPE>
  void test_gather_out() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;
    const std::vector<int32_t> sizes = {2, 3};
    // clang-format off
    Tensor self = tf_data.make(
      /*sizes=*/{2, 5},
      {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
      });
    // clang-format on
    Tensor out = tf_data.zeros(sizes);
    // clang-format off
    bool sparse_grad = false;
    Tensor index = tf_index.make(sizes,
      {
        0, 1, 0,
        1, 0, 1,
      });
    // clang-format on

    // Valid input should give the expected output
    op_gather_out(self, 0, index, sparse_grad, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(
          sizes,
          {
            1, 7, 3,
            6, 2, 8,
          }));
    // clang-format on

    // Valid input should give the expected output
    op_gather_out(self, 1, index, sparse_grad, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(sizes,
        {
          1, 2, 1,
          7, 6, 7,
        }));

    self = tf_data.make(
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
    index = tf_index.make(
      /*sizes=*/{1, 3, 2},
      {
        0, 1,
        1, 2,
        0, 2
      });
    // clang-format on
    out = tf_data.zeros(/*sizes=*/{1, 3, 2});

    op_gather_out(self, 1, index, sparse_grad, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{1, 3, 2},
            {
              1, 5,
              4, 8,
              1, 8,
            }));
    // clang-format on

    out = tf_data.zeros(/*sizes=*/{1, 3, 2});
    op_gather_out(self, 2, index, sparse_grad, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{1, 3, 2},
            {
              1, 2,
              5, 6,
              7, 9,
            }));
    // clang-format on
  }

  // Invalid dimensions
  template <ScalarType DATA_DTYPE>
  void test_gather_out_invalid_dim() {
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
    bool sparse_grad = false;
    Tensor out = tf_data.zeros(sizes);

    // Invalid dim should die
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_gather_out(self, -3, index, sparse_grad, out));
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_gather_out(self, 2, index, sparse_grad, out));

    // Self and index hsould have same number of dimensions
    index = tf_index.zeros(/*sizes=*/{2, 2, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_gather_out(self, 0, index, sparse_grad, out));

    // Size of dimension of index should be smaller than the size of that
    // dimension of self if dimension != dim
    index = tf_index.zeros(/*sizes=*/{3, 5});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_gather_out(self, 1, index, sparse_grad, out));

    // Index out of bound for self in dim
    index = tf_index.make(/*sizes=*/{2, 3}, {0, 1, 2, 0, 1, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_gather_out(self, 0, index, sparse_grad, out));
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    TensorFactory<ScalarType::Int> tf;
    TensorFactory<ScalarType::Long> tf_index;

    Tensor input = tf.ones({2, 3, 4});
    Tensor index = tf_index.zeros({2, 3, 4});
    bool sparse_grad = false;
    Tensor expected = tf.ones({2, 3, 4});
    Tensor out = tf.zeros(out_shape, dynamism);

    op_gather_out(input, 2, index, sparse_grad, out);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpGatherOutTest, AllValidInputOutputSupport) {
#define TEST_ENTRY(CTYPE, DTYPE) test_gather_out<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGatherOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;
  // clang-format off
  Tensor self = tf_data.make(
      /*sizes=*/{2, 5},
      {
        INFINITY, -INFINITY, NAN,       2.33, 3.14,
        NAN,      INFINITY,  -INFINITY, 3.14, 2.33
      });
  // clang-format on
  const std::vector<int32_t> sizes = {2, 3};
  Tensor index = tf_index.make(sizes, {0, 1, 0, 1, 0, 1});
  bool sparse_grad = false;
  Tensor out = tf_data.zeros(sizes);

  // Valid input should give the expected output
  op_gather_out(self, 0, index, sparse_grad, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(
      out,
      tf_data.make(sizes,
      {
        INFINITY, INFINITY, NAN,
        NAN, -INFINITY, -INFINITY,
      }));
  // clang-format on
}

TEST_F(OpGatherOutTest, InvalidDimensionsDies) {
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_gather_out_invalid_dim<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGatherOutTest, MismatchedInputDtypesDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor self = tf_char.make({2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  const std::vector<int32_t> sizes = {2, 3};
  Tensor index = tf_byte.make(sizes, {0, 1, 0, 0, 1, 0});
  bool sparse_grad = false;
  Tensor out = tf_char.zeros(sizes);

  // Types other than long for index should die
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_gather_out(self, 0, index, sparse_grad, out));

  // Mismatched dtype of self and out should die
  self = tf_byte.make(/*sizes=*/{2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  index = tf_long.make(sizes, {0, 1, 0, 1, 0, 1});
  out = tf_char.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_gather_out(self, 0, index, sparse_grad, out));
}

TEST_F(OpGatherOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpGatherOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpGatherOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}

TEST_F(OpGatherOutTest, EmptyIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.ones({2, 5});
  const std::vector<int32_t> sizes = {2, 0, 3};
  Tensor index = tf_index.zeros(sizes);
  bool sparse_grad = false;
  Tensor out = tf_data.zeros(sizes);
  op_gather_out(self, 0, index, sparse_grad, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.zeros(sizes));
}

TEST_F(OpGatherOutTest, ValidZeroDim) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  Tensor index = tf_index.zeros({});
  bool sparse_grad = false;
  Tensor out = tf_data.zeros({});
  op_gather_out(self, 0, index, sparse_grad, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({}, {3.14}));
}

TEST_F(OpGatherOutTest, InvalidZeroDimInput) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.ones({});
  const std::vector<int32_t> sizes = {2, 3};
  Tensor index = tf_index.make(sizes, {0, 0, 0, 0, 0, 0});
  bool sparse_grad = false;
  Tensor out = tf_data.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_gather_out(self, 0, index, sparse_grad, out));
}

TEST_F(OpGatherOutTest, InvalidZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({2, 3}, {1, 2, 3, 4, 5, 6});
  const std::vector<int32_t> sizes = {};
  Tensor index = tf_index.make(sizes, {2});
  bool sparse_grad = false;
  Tensor out = tf_data.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_gather_out(self, 1, index, sparse_grad, out));
}

TEST_F(OpGatherOutTest, ValidZeroDimInputAndOneDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  const std::vector<int32_t> sizes = {3};
  Tensor index = tf_index.make(sizes, {0, 0, 0});
  bool sparse_grad = false;
  Tensor out = tf_data.make({3}, {2.71, 2.71, 2.71});
  op_gather_out(self, 0, index, sparse_grad, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({3}, {3.14, 3.14, 3.14}));
}

TEST_F(OpGatherOutTest, ValidOneDimInputAndZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({3}, {10, 20, 30});
  const std::vector<int32_t> sizes = {};
  Tensor index = tf_index.make(sizes, {2});
  bool sparse_grad = false;
  Tensor out = tf_data.make(sizes, {1729});
  op_gather_out(self, 0, index, sparse_grad, out);
  EXPECT_TENSOR_CLOSE(out, tf_data.make({}, {30}));
}

TEST_F(OpGatherOutTest, InvalidZeroDimInputAndOneDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({}, {3.14});
  const std::vector<int32_t> sizes = {3};
  Tensor index = tf_index.make(sizes, {10, 100, 1000});
  bool sparse_grad = false;
  Tensor out = tf_data.make({3}, {2.71, 2.71, 2.71});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_gather_out(self, 0, index, sparse_grad, out));
}

TEST_F(OpGatherOutTest, InvalidOneDimInputAndZeroDimIndex) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;

  Tensor self = tf_data.make({3}, {10, 20, 30});
  const std::vector<int32_t> sizes = {};
  Tensor index = tf_index.make(sizes, {100});
  bool sparse_grad = false;
  Tensor out = tf_data.make(sizes, {1729});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_gather_out(self, 0, index, sparse_grad, out));
}
