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

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext;
using torch::executor::testing::TensorFactory;

class OpLeScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_le_scalar_out(const Tensor& self, Scalar& other, Tensor& out) {
    return torch::executor::aten::le_outf(context_, self, other, out);
  }

  template <ScalarType DTYPE_IN, ScalarType DTYPE_OUT>
  void test_le_scalar_out() {
    TensorFactory<DTYPE_IN> tf;
    TensorFactory<DTYPE_OUT> tf_out;

    const std::vector<int32_t> sizes = {2, 2};
    Tensor out = tf_out.ones(sizes);
    Scalar other = 2;

    // Valid input should give the expected output
    op_le_scalar_out(tf.make(sizes, /*data=*/{3, 1, 2, 4}), other, out);
    EXPECT_TENSOR_EQ(
        out, tf_out.make(sizes, /*data=*/{false, true, true, false}));
  }
};

class OpLeTensorOutTest : public OperatorTest {
 protected:
  Tensor&
  op_le_tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::aten::le_outf(context_, self, other, out);
  }

  template <ScalarType DTYPE_IN, ScalarType DTYPE_OUT>
  void test_dtype() {
    TensorFactory<DTYPE_IN> tf_input;
    TensorFactory<DTYPE_OUT> tf_out;
    Tensor a = tf_input.make(/*sizes=*/{2, 2}, /*data=*/{2, 3, 2, 4});
    Tensor b = tf_input.make({2, 2}, {1, 4, 2, 3});
    Tensor out = tf_out.zeros({2, 2});

    op_le_tensor_out(a, b, out);
    EXPECT_TENSOR_EQ(out, tf_out.make({2, 2}, {false, true, true, false}));
  }
};

TEST_F(OpLeScalarOutTest, AllRealInputBoolOutputSupport) {
#define TEST_ENTRY(ctype_in, dtype_in, ctype_out, dtype_out) \
  test_le_scalar_out<ScalarType::dtype_in, ScalarType::dtype_out>();

#define TEST_FORALL_OUT_TYPES(ctype_in, dtype_in)                 \
  ET_FORALL_REALHBF16_TYPES_WITH2(ctype_in, dtype_in, TEST_ENTRY) \
  test_le_scalar_out<ScalarType::dtype_in, ScalarType::Bool>();

  ET_FORALL_REALHBF16_TYPES(TEST_FORALL_OUT_TYPES)

#undef TEST_FORALL_OUT_TYPES
#undef TEST_ENTRY
}

TEST_F(OpLeScalarOutTest, BoolInputDtype) {
  TensorFactory<ScalarType::Bool> tf_bool;

  const std::vector<int32_t> sizes = {2, 2};
  Tensor a = tf_bool.make(sizes, /*data=*/{false, true, false, true});
  Tensor out = tf_bool.zeros(sizes);
  Scalar other = 0.5;

  op_le_scalar_out(a, other, out);
  EXPECT_TENSOR_EQ(
      out, tf_bool.make(sizes, /*data=*/{true, false, true, false}));
}

// Mismatched shape tests.
TEST_F(OpLeScalarOutTest, MismatchedInOutShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_bool.ones(/*sizes=*/{2, 2});
  Scalar other = 3;

  ET_EXPECT_KERNEL_FAILURE(context_, op_le_scalar_out(a, other, out));
}

TEST_F(OpLeScalarOutTest, DynamicOutShapeTest) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};
  const std::vector<int32_t> out_sizes = {4, 1};

  Tensor out =
      tf.zeros(out_sizes, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Scalar other = 2;

  // Valid input should give the expected output
  op_le_scalar_out(tf.make(sizes, /*data=*/{3, 1, 2, 4}), other, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{false, true, true, false}));
}

TEST_F(OpLeTensorOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype_in, dtype_in, ctype_out, dtype_out) \
  test_dtype<ScalarType::dtype_in, ScalarType::dtype_out>();

#define TEST_FORALL_OUT_TYPES(ctype_in, dtype_in)                 \
  ET_FORALL_REALHBF16_TYPES_WITH2(ctype_in, dtype_in, TEST_ENTRY) \
  test_dtype<ScalarType::dtype_in, ScalarType::Bool>();

  ET_FORALL_REALHBF16_TYPES(TEST_FORALL_OUT_TYPES);

#undef TEST_FORALL_OUT_TYPES
#undef TEST_ENTRY
}

TEST_F(OpLeTensorOutTest, MismatchedInShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor b = tf_int.ones(/*sizes=*/{2, 2});
  Tensor out = tf_bool.ones(/*sizes=*/{4});

  ET_EXPECT_KERNEL_FAILURE(context_, op_le_tensor_out(a, b, out));
}

TEST_F(OpLeTensorOutTest, MismatchedInOutShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor b = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_bool.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_le_tensor_out(a, b, out));
}

TEST_F(OpLeTensorOutTest, DynamicOutShapeTest) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{2, 3, 2, 4});
  Tensor b = tf.make({2, 2}, {1, 4, 2, 3});

  Tensor out =
      tf.zeros({1, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  op_le_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {false, true, true, false}));
}

TEST_F(OpLeTensorOutTest, BroadcastTest) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make(/*sizes=*/{4}, /*data=*/{2, 3, 2, 4});
  Tensor b = tf.make({1, 1}, {3});

  Tensor out = tf.zeros({1, 4});

  op_le_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({1, 4}, {true, true, true, false}));
}

TEST_F(OpLeTensorOutTest, Broadcast2DTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case: (1, 10) and (6, 1) -> (6, 10)
  Tensor a =
      tf.make(/*sizes=*/{1, 10}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  Tensor b = tf.make({6, 1}, {2, 4, 6, 8, 10, 12});

  Tensor out = tf_bool.zeros({6, 10});

  op_le_tensor_out(a, b, out);

  // Expected: each row i should be [1<=b[i], 2<=b[i], ..., 10<=b[i]]
  // Row 0: b[0]=2, so [1<=2, 2<=2, 3<=2, ...] = [true, true, false, false, ...]
  // Row 1: b[1]=4, so [1<=4, 2<=4, 3<=4, 4<=4, 5<=4, ...] = [true, true, true,
  // true, false, ...]
  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // Row 0 (b=2): 1<=2, 2<=2, 3<=2, 4<=2, 5<=2, 6<=2, 7<=2, 8<=2, 9<=2,
      // 10<=2
      true,
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      // Row 1 (b=4): 1<=4, 2<=4, 3<=4, 4<=4, 5<=4, 6<=4, 7<=4, 8<=4, 9<=4,
      // 10<=4
      true,
      true,
      true,
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      // Row 2 (b=6): 1<=6, 2<=6, 3<=6, 4<=6, 5<=6, 6<=6, 7<=6, 8<=6, 9<=6,
      // 10<=6
      true,
      true,
      true,
      true,
      true,
      true,
      false,
      false,
      false,
      false,
      // Row 3 (b=8): 1<=8, 2<=8, 3<=8, 4<=8, 5<=8, 6<=8, 7<=8, 8<=8, 9<=8,
      // 10<=8
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      false,
      false,
      // Row 4 (b=10): 1<=10, 2<=10, 3<=10, 4<=10, 5<=10, 6<=10, 7<=10, 8<=10,
      // 9<=10, 10<=10
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      // Row 5 (b=12): 1<=12, 2<=12, 3<=12, 4<=12, 5<=12, 6<=12, 7<=12, 8<=12,
      // 9<=12, 10<=12
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true};

  EXPECT_TENSOR_EQ(out, tf_bool.make({6, 10}, expected_data));
}

TEST_F(OpLeTensorOutTest, Broadcast1DTo2DTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case: (6,) and (1, 10) -> (6, 10)
  Tensor a = tf.make({6, 1}, {2, 4, 6, 8, 10, 12});
  Tensor b =
      tf.make(/*sizes=*/{1, 10}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  Tensor out = tf_bool.zeros({6, 10});

  op_le_tensor_out(a, b, out);

  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // Row 0 (a=2): 2<=1, 2<=2, 2<=3, 2<=4, 2<=5, 2<=6, 2<=7, 2<=8, 2<=9,
      // 2<=10
      false,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      // Row 1 (a=4): 4<=1, 4<=2, 4<=3, 4<=4, 4<=5, 4<=6, 4<=7, 4<=8, 4<=9,
      // 4<=10
      false,
      false,
      false,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      // Row 2 (a=6): 6<=1, 6<=2, 6<=3, 6<=4, 6<=5, 6<=6, 6<=7, 6<=8, 6<=9,
      // 6<=10
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      true,
      true,
      true,
      // Row 3 (a=8): 8<=1, 8<=2, 8<=3, 8<=4, 8<=5, 8<=6, 8<=7, 8<=8, 8<=9,
      // 8<=10
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      true,
      // Row 4 (a=10): 10<=1, 10<=2, 10<=3, 10<=4, 10<=5, 10<=6, 10<=7, 10<=8,
      // 10<=9, 10<=10
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      true,
      // Row 5 (a=12): 12<=1, 12<=2, 12<=3, 12<=4, 12<=5, 12<=6, 12<=7, 12<=8,
      // 12<=9, 12<=10
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false};

  EXPECT_TENSOR_EQ(out, tf_bool.make({6, 10}, expected_data));
}

TEST_F(OpLeTensorOutTest, BroadcastReverseTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case: (6, 1) and (1, 10) -> (6, 10) (reverse of the first broadcast
  // test)
  Tensor a = tf.make(/*sizes=*/{6, 1}, /*data=*/{2, 4, 6, 8, 10, 12});
  Tensor b = tf.make({1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  Tensor out = tf_bool.zeros({6, 10});

  op_le_tensor_out(a, b, out);

  // Expected: each row i should be [a[i]<=1, a[i]<=2, ..., a[i]<=10]
  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // Row 0 (a=2): 2<=1, 2<=2, 2<=3, 2<=4, 2<=5, 2<=6, 2<=7, 2<=8, 2<=9,
      // 2<=10
      false,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      // Row 1 (a=4): 4<=1, 4<=2, 4<=3, 4<=4, 4<=5, 4<=6, 4<=7, 4<=8, 4<=9,
      // 4<=10
      false,
      false,
      false,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      // Row 2 (a=6): 6<=1, 6<=2, 6<=3, 6<=4, 6<=5, 6<=6, 6<=7, 6<=8, 6<=9,
      // 6<=10
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      true,
      true,
      true,
      // Row 3 (a=8): 8<=1, 8<=2, 8<=3, 8<=4, 8<=5, 8<=6, 8<=7, 8<=8, 8<=9,
      // 8<=10
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      true,
      // Row 4 (a=10): 10<=1, 10<=2, 10<=3, 10<=4, 10<=5, 10<=6, 10<=7, 10<=8,
      // 10<=9, 10<=10
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      true,
      // Row 5 (a=12): 12<=1, 12<=2, 12<=3, 12<=4, 12<=5, 12<=6, 12<=7, 12<=8,
      // 12<=9, 12<=10
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false};

  EXPECT_TENSOR_EQ(out, tf_bool.make({6, 10}, expected_data));
}

TEST_F(OpLeTensorOutTest, BroadcastLastDimTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case for kBroadcastLastDim: (3, 4, 1) and (3, 4, 5) -> (3, 4, 5)
  Tensor a = tf.make(
      /*sizes=*/{3, 4, 1}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor b = tf.make(
      {3, 4, 5},
      {
          // First 3x4 slice
          1,
          2,
          3,
          4,
          5, // row 0
          2,
          3,
          4,
          5,
          6, // row 1
          3,
          4,
          5,
          6,
          7, // row 2
          4,
          5,
          6,
          7,
          8, // row 3
             // Second 3x4 slice
          5,
          6,
          7,
          8,
          9, // row 0
          6,
          7,
          8,
          9,
          10, // row 1
          7,
          8,
          9,
          10,
          11, // row 2
          8,
          9,
          10,
          11,
          12, // row 3
              // Third 3x4 slice
          9,
          10,
          11,
          12,
          13, // row 0
          10,
          11,
          12,
          13,
          14, // row 1
          11,
          12,
          13,
          14,
          15, // row 2
          12,
          13,
          14,
          15,
          16 // row 3
      });

  Tensor out = tf_bool.zeros({3, 4, 5});

  op_le_tensor_out(a, b, out);

  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // First slice: a values are 1,2,3,4
      true,
      true,
      true,
      true,
      true, // 1 <= [1,2,3,4,5]
      true,
      true,
      true,
      true,
      true, // 2 <= [2,3,4,5,6]
      true,
      true,
      true,
      true,
      true, // 3 <= [3,4,5,6,7]
      true,
      true,
      true,
      true,
      true, // 4 <= [4,5,6,7,8]
      // Second slice: a values are 5,6,7,8
      true,
      true,
      true,
      true,
      true, // 5 <= [5,6,7,8,9]
      true,
      true,
      true,
      true,
      true, // 6 <= [6,7,8,9,10]
      true,
      true,
      true,
      true,
      true, // 7 <= [7,8,9,10,11]
      true,
      true,
      true,
      true,
      true, // 8 <= [8,9,10,11,12]
      // Third slice: a values are 9,10,11,12
      true,
      true,
      true,
      true,
      true, // 9 <= [9,10,11,12,13]
      true,
      true,
      true,
      true,
      true, // 10 <= [10,11,12,13,14]
      true,
      true,
      true,
      true,
      true, // 11 <= [11,12,13,14,15]
      true,
      true,
      true,
      true,
      true // 12 <= [12,13,14,15,16]
  };

  EXPECT_TENSOR_EQ(out, tf_bool.make({3, 4, 5}, expected_data));
}

TEST_F(OpLeTensorOutTest, BroadcastLastDimReverseTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case for kBroadcastLastDimReverseArguments: (3, 4, 5) and (3, 4, 1) ->
  // (3, 4, 5)
  Tensor a = tf.make(
      {3, 4, 5},
      {
          // First 3x4 slice
          1,
          2,
          3,
          4,
          5, // row 0
          2,
          3,
          4,
          5,
          6, // row 1
          3,
          4,
          5,
          6,
          7, // row 2
          4,
          5,
          6,
          7,
          8, // row 3
             // Second 3x4 slice
          5,
          6,
          7,
          8,
          9, // row 0
          6,
          7,
          8,
          9,
          10, // row 1
          7,
          8,
          9,
          10,
          11, // row 2
          8,
          9,
          10,
          11,
          12, // row 3
              // Third 3x4 slice
          9,
          10,
          11,
          12,
          13, // row 0
          10,
          11,
          12,
          13,
          14, // row 1
          11,
          12,
          13,
          14,
          15, // row 2
          12,
          13,
          14,
          15,
          16 // row 3
      });
  Tensor b = tf.make(
      /*sizes=*/{3, 4, 1},
      /*data=*/{5, 5, 5, 5, 10, 10, 10, 10, 15, 15, 15, 15});

  Tensor out = tf_bool.zeros({3, 4, 5});

  op_le_tensor_out(a, b, out);

  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // First slice: b values are all 5
      true,
      true,
      true,
      true,
      true, // [1,2,3,4,5] <= 5
      true,
      true,
      true,
      true,
      false, // [2,3,4,5,6] <= 5
      true,
      true,
      true,
      false,
      false, // [3,4,5,6,7] <= 5
      true,
      true,
      false,
      false,
      false, // [4,5,6,7,8] <= 5
      // Second slice: b values are all 10
      true,
      true,
      true,
      true,
      true, // [5,6,7,8,9] <= 10
      true,
      true,
      true,
      true,
      true, // [6,7,8,9,10] <= 10
      true,
      true,
      true,
      true,
      false, // [7,8,9,10,11] <= 10
      true,
      true,
      true,
      false,
      false, // [8,9,10,11,12] <= 10
      // Third slice: b values are all 15
      true,
      true,
      true,
      true,
      true, // [9,10,11,12,13] <= 15
      true,
      true,
      true,
      true,
      true, // [10,11,12,13,14] <= 15
      true,
      true,
      true,
      true,
      true, // [11,12,13,14,15] <= 15
      true,
      true,
      true,
      true,
      false // [12,13,14,15,16] <= 15
  };

  EXPECT_TENSOR_EQ(out, tf_bool.make({3, 4, 5}, expected_data));
}

TEST_F(OpLeTensorOutTest, BroadcastNdByNdTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case for kBroadcastNdByNd: (2, 1, 4) and (2, 3, 4) -> (2, 3, 4)
  Tensor a = tf.make(/*sizes=*/{2, 1, 4}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8});
  Tensor b = tf.make(
      {2, 3, 4},
      {
          // First 2x3 slice
          1,
          2,
          3,
          4, // row 0
          2,
          3,
          4,
          5, // row 1
          3,
          4,
          5,
          6, // row 2
             // Second 2x3 slice
          5,
          6,
          7,
          8, // row 0
          6,
          7,
          8,
          9, // row 1
          7,
          8,
          9,
          10 // row 2
      });

  Tensor out = tf_bool.zeros({2, 3, 4});

  op_le_tensor_out(a, b, out);

  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // First slice: a[0,0,:] = [1,2,3,4]
      true,
      true,
      true,
      true, // [1,2,3,4] <= [1,2,3,4]
      true,
      true,
      true,
      true, // [1,2,3,4] <= [2,3,4,5]
      true,
      true,
      true,
      true, // [1,2,3,4] <= [3,4,5,6]
      // Second slice: a[1,0,:] = [5,6,7,8]
      true,
      true,
      true,
      true, // [5,6,7,8] <= [5,6,7,8]
      true,
      true,
      true,
      true, // [5,6,7,8] <= [6,7,8,9]
      true,
      true,
      true,
      true // [5,6,7,8] <= [7,8,9,10]
  };

  EXPECT_TENSOR_EQ(out, tf_bool.make({2, 3, 4}, expected_data));
}

TEST_F(OpLeTensorOutTest, BroadcastNdByNdReverseTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case for kBroadcastNdByNdReverseArguments: (2, 3, 4) and (2, 1, 4) ->
  // (2, 3, 4)
  Tensor a = tf.make(
      {2, 3, 4},
      {
          // First 2x3 slice
          1,
          2,
          3,
          4, // row 0
          2,
          3,
          4,
          5, // row 1
          3,
          4,
          5,
          6, // row 2
             // Second 2x3 slice
          5,
          6,
          7,
          8, // row 0
          6,
          7,
          8,
          9, // row 1
          7,
          8,
          9,
          10 // row 2
      });
  Tensor b = tf.make(/*sizes=*/{2, 1, 4}, /*data=*/{2, 3, 4, 5, 6, 7, 8, 9});

  Tensor out = tf_bool.zeros({2, 3, 4});

  op_le_tensor_out(a, b, out);

  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // First slice: b[0,0,:] = [2,3,4,5]
      true,
      true,
      true,
      true, // [1,2,3,4] <= [2,3,4,5]
      true,
      true,
      true,
      true, // [2,3,4,5] <= [2,3,4,5]
      false,
      false,
      false,
      false, // [3,4,5,6] <= [2,3,4,5]
      // Second slice: b[1,0,:] = [6,7,8,9]
      true,
      true,
      true,
      true, // [5,6,7,8] <= [6,7,8,9]
      true,
      true,
      true,
      true, // [6,7,8,9] <= [6,7,8,9]
      false,
      false,
      false,
      false // [7,8,9,10] <= [6,7,8,9]
  };

  EXPECT_TENSOR_EQ(out, tf_bool.make({2, 3, 4}, expected_data));
}

TEST_F(OpLeTensorOutTest, Broadcast2dBy1dTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case for kBroadcast2dBy1d: (3, 4) and (4,) -> (3, 4)
  Tensor a = tf.make(
      /*sizes=*/{3, 4}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor b = tf.make({4}, {2, 4, 6, 8});

  Tensor out = tf_bool.zeros({3, 4});

  op_le_tensor_out(a, b, out);

  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      true,
      true,
      true,
      true, // [1,2,3,4] <= [2,4,6,8]
      false,
      false,
      false,
      true, // [5,6,7,8] <= [2,4,6,8]
      false,
      false,
      false,
      false // [9,10,11,12] <= [2,4,6,8]
  };

  EXPECT_TENSOR_EQ(out, tf_bool.make({3, 4}, expected_data));
}

TEST_F(OpLeTensorOutTest, Broadcast1DTo2DShapeTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case: (6,) and (1, 6) -> (1, 6)
  Tensor a = tf.make({6}, {1, 3, 5, 7, 9, 11});
  Tensor b = tf.make({1, 6}, {2, 4, 6, 8, 10, 12});

  Tensor out = tf_bool.zeros({1, 6});

  op_le_tensor_out(a, b, out);

  // Expected: a[i] <= b[0,i] for all i
  // [1, 3, 5, 7, 9, 11] <= [2, 4, 6, 8, 10, 12]
  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      true, // 1 <= 2
      true, // 3 <= 4
      true, // 5 <= 6
      true, // 7 <= 8
      true, // 9 <= 10
      true // 11 <= 12
  };

  EXPECT_TENSOR_EQ(out, tf_bool.make({1, 6}, expected_data));
}

TEST_F(OpLeTensorOutTest, Broadcast2DBy1DShapeTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case: (10,) and (6, 1) -> (6, 10)
  Tensor a = tf.make({10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  Tensor b = tf.make({6, 1}, {2, 4, 6, 8, 10, 12});

  Tensor out = tf_bool.zeros({6, 10});

  op_le_tensor_out(a, b, out);

  // Expected: a[j] <= b[i,0] for all i,j
  // Each row i should be [a[0]<=b[i,0], a[1]<=b[i,0], ..., a[9]<=b[i,0]]
  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      // Row 0 (b=2): [1,2,3,4,5,6,7,8,9,10] <= 2
      true,
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      // Row 1 (b=4): [1,2,3,4,5,6,7,8,9,10] <= 4
      true,
      true,
      true,
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      // Row 2 (b=6): [1,2,3,4,5,6,7,8,9,10] <= 6
      true,
      true,
      true,
      true,
      true,
      true,
      false,
      false,
      false,
      false,
      // Row 3 (b=8): [1,2,3,4,5,6,7,8,9,10] <= 8
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      false,
      false,
      // Row 4 (b=10): [1,2,3,4,5,6,7,8,9,10] <= 10
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      // Row 5 (b=12): [1,2,3,4,5,6,7,8,9,10] <= 12
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true};

  EXPECT_TENSOR_EQ(out, tf_bool.make({6, 10}, expected_data));
}

TEST_F(OpLeTensorOutTest, Broadcast22dBy1dReverseTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case for kBroadcast2dBy1dReverseArguments: (4,) and (3, 4) -> (3, 4)
  Tensor a = tf.make({4}, {2, 4, 6, 8});
  Tensor b = tf.make(
      /*sizes=*/{3, 4}, /*data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  Tensor out = tf_bool.zeros({3, 4});

  op_le_tensor_out(a, b, out);

  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data = {
      false,
      false,
      false,
      false, // [2,4,6,8] <= [1,2,3,4]
      true,
      true,
      true,
      true, // [2,4,6,8] <= [5,6,7,8]
      true,
      true,
      true,
      true // [2,4,6,8] <= [9,10,11,12]
  };

  EXPECT_TENSOR_EQ(out, tf_bool.make({3, 4}, expected_data));
}

TEST_F(OpLeTensorOutTest, MonotonicIncreasingVsScalarBroadcastTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Test case: 1D tensor [0, 1, 2, ..., 63] vs 2D tensor [1, 1] with value 2
  std::vector<int32_t> lhs_data;
  for (int i = 0; i < 64; ++i) {
    lhs_data.push_back(i);
  }

  Tensor lhs = tf.make({64}, lhs_data);
  Tensor rhs = tf.make({1, 1}, {2});
  Tensor out = tf_bool.zeros({1, 64});

  op_le_tensor_out(lhs, rhs, out);

  // Expected: [0, 1, 2] <= 2 should be [true, true, true], rest false
  using ctype =
      executorch::runtime::testing::internal::ScalarTypeToCppTypeWrapper<
          ScalarType::Bool>::ctype;
  std::vector<ctype> expected_data;
  for (int i = 0; i < 64; ++i) {
    expected_data.push_back(i <= 2);
  }

  EXPECT_TENSOR_EQ(out, tf_bool.make({1, 64}, expected_data));

  // Test with rhs value 4
  rhs = tf.make({1, 1}, {4});
  out = tf_bool.zeros({1, 64});

  op_le_tensor_out(lhs, rhs, out);

  expected_data.clear();
  for (int i = 0; i < 64; ++i) {
    expected_data.push_back(i <= 4);
  }

  EXPECT_TENSOR_EQ(out, tf_bool.make({1, 64}, expected_data));

  // Test with rhs value 10
  rhs = tf.make({1, 1}, {10});
  out = tf_bool.zeros({1, 64});

  op_le_tensor_out(lhs, rhs, out);

  expected_data.clear();
  for (int i = 0; i < 64; ++i) {
    expected_data.push_back(i <= 10);
  }

  EXPECT_TENSOR_EQ(out, tf_bool.make({1, 64}, expected_data));

  // Test with rhs value 32
  rhs = tf.make({1, 1}, {32});
  out = tf_bool.zeros({1, 64});

  op_le_tensor_out(lhs, rhs, out);

  expected_data.clear();
  for (int i = 0; i < 64; ++i) {
    expected_data.push_back(i <= 32);
  }

  EXPECT_TENSOR_EQ(out, tf_bool.make({1, 64}, expected_data));
}
