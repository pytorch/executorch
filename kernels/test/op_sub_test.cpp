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
using executorch::runtime::testing::TensorFactory;
using torch::executor::testing::SupportedFeatures;
namespace etrt = executorch::runtime;

class OpSubOutTest : public OperatorTest {
 protected:
  Tensor& op_sub_out(
      const Tensor& self,
      const Tensor& other,
      const Scalar& alpha,
      Tensor& out) {
    return torch::executor::aten::sub_outf(context_, self, other, alpha, out);
  }

  template <ScalarType DTYPE_A, ScalarType DTYPE_B, ScalarType DTYPE_OUT>
  void test_sub() {
    TensorFactory<DTYPE_A> tf_a;
    TensorFactory<DTYPE_B> tf_b;
    TensorFactory<DTYPE_OUT> tf_out;

    const std::vector<int32_t> sizes = {2, 2};

    // Destination for the sum.
    Tensor out = tf_out.zeros(sizes);

    // sub two tensors.
    op_sub_out(
        tf_a.make(sizes, /*data=*/{1, 2, 4, 8}),
        tf_b.ones(sizes),
        /*alpha=*/1,
        out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_EQ(out, tf_out.make(sizes, /*data=*/{0, 1, 3, 7}));
  }

  template <ScalarType DTYPE_A, ScalarType DTYPE_B>
  void test_sub_enumerate_out_types() {
    test_sub<DTYPE_A, DTYPE_B, ScalarType::Half>();
    test_sub<DTYPE_A, DTYPE_B, ScalarType::Float>();
    test_sub<DTYPE_A, DTYPE_B, ScalarType::Double>();
    // Integral out type is only allowed if both inputs are integral types
    if (etrt::isIntegralType(DTYPE_A, false) &&
        etrt::isIntegralType(DTYPE_B, false)) {
      test_sub<DTYPE_A, DTYPE_B, ScalarType::Int>();
      test_sub<DTYPE_A, DTYPE_B, ScalarType::Long>();
    }
  }

  template <ScalarType DTYPE_A>
  void test_sub_enumerate_b_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_sub_enumerate_out_types<DTYPE_A, ScalarType::dtype>();

    ET_FORALL_REAL_TYPES_AND(Half, ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  // Common testing for substraction between two floating point Tensors.
  template <ScalarType DTYPE>
  void test_floating_point_sub_out() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 2};

    // Destination for the subtraction.
    Tensor out = tf.zeros(sizes);

    // Performs substraction on two tensors.
    op_sub_out(
        tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}),
        tf.ones(sizes),
        /*alpha=*/1,
        out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{0.1, 1.2, 3.4, 7.8}));
  }

  void test_sub_enumerate_a_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_sub_enumerate_b_types<ScalarType::dtype>();

    ET_FORALL_REAL_TYPES_AND(Half, ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  template <ScalarType DTYPE>
  void test_broadcast_rank1_scalar() {
    TensorFactory<DTYPE> tf;

    Tensor a = tf.make({2, 1, 3}, {2, 3, 4, 5, 6, 7});
    Tensor b = tf.make({1}, {2});

    // Destination for the broadcasting div. Follow the broadcasting rules in
    // https://fburl.com/n9wl4d0o
    Tensor out = tf.zeros({2, 1, 3});

    op_sub_out(a, b, 1, out);

    Tensor ret = tf.make({2, 1, 3}, {0, 1, 2, 3, 4, 5});
    EXPECT_TENSOR_EQ(out, ret);

    op_sub_out(b, a, 1, out);
    ret = tf.make({2, 1, 3}, {0, -1, -2, -3, -4, -5});
    EXPECT_TENSOR_EQ(out, ret);
  }
};

class OpSubScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_sub_scalar_out(
      const Tensor& self,
      const Scalar& other,
      const Scalar& alpha,
      Tensor& out) {
    return torch::executor::aten::sub_outf(context_, self, other, alpha, out);
  }
};

/**
 * Uses the function templates above to test all valid combinations of inputs
 * and output dtypes
 */
TEST_F(OpSubOutTest, AllRealDtypesSupported) {
  test_sub_enumerate_a_types();
}

TEST_F(OpSubOutTest, FloatTensors) {
  test_floating_point_sub_out<ScalarType::Float>();
}

TEST_F(OpSubOutTest, DoubleTensors) {
  test_floating_point_sub_out<ScalarType::Double>();
}

TEST_F(OpSubOutTest, BroadcastSupported) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({2, 1, 2, 1}, {7, 8, 9, 10});
  Tensor b = tf.make({2, 1, 4}, {1, 1, 1, 1, 2, 2, 2, 2});
  Tensor ref =
      tf.make({2, 2, 2, 4}, {6, 6, 6, 6, 7, 7, 7, 7, 5, 5, 5, 5, 6, 6, 6, 6,
                             8, 8, 8, 8, 9, 9, 9, 9, 7, 7, 7, 7, 8, 8, 8, 8});

  // Destination for the broadcasting sum. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({2, 2, 2, 4});

  op_sub_out(a, b, 1, out);

  EXPECT_TENSOR_EQ(out, ref);
}

TEST_F(OpSubOutTest, BroadcastSupported2) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({3, 2, 1}, {2, 3, 4, 5, 6, 7});
  Tensor b = tf.make({1, 2, 1}, {2, 3});

  // Destination for the broadcasting div. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({3, 2, 1});

  op_sub_out(a, b, 1, out);

  Tensor ret = tf.make({3, 2, 1}, {0, 0, 2, 2, 4, 4});
  EXPECT_TENSOR_EQ(out, ret);
}

TEST_F(OpSubOutTest, BroadcastScalarSupported1) {
  test_broadcast_rank1_scalar<ScalarType::Float>();
  test_broadcast_rank1_scalar<ScalarType::Half>();
}

TEST_F(OpSubOutTest, BroadcastScalarSupported2) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({1, 1, 1}, {8});
  Tensor b = tf.make({3, 1, 1}, {2, 4, 8});

  // Destination for the broadcasting div. Follow the broadcasting rules in
  // https://fburl.com/n9wl4d0o
  Tensor out = tf.zeros({3, 1, 1});

  op_sub_out(a, b, 1, out);

  Tensor ret = tf.make({3, 1, 1}, {6, 4, 0});
  EXPECT_TENSOR_EQ(out, ret);

  std::swap(a, b);
  out = tf.zeros({3, 1, 1});
  op_sub_out(a, b, 1, out);
  ret = tf.make({3, 1, 1}, {-6, -4, 0});
  EXPECT_TENSOR_EQ(out, ret);
}

TEST_F(OpSubOutTest, BroadcastScalarRank0Supported) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.make({1}, {5});
  Tensor b = tf.make({}, {2});

  Tensor out = tf.zeros({1});

  op_sub_out(a, b, 1, out);

  Tensor ret = tf.make({1}, {3});
  EXPECT_TENSOR_EQ(out, ret);

  op_sub_out(b, a, 1, out);

  ret = tf.make({1}, {-3});
  EXPECT_TENSOR_EQ(out, ret);
}

//
// Death Tests
//

TEST_F(OpSubOutTest, IntTensorFloatAlphaDies) {
  // op_sub_out() doesn't handle floating alpha for intergal inputs
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the op.
  Tensor out = tf.zeros(sizes);

  // Subtraction operation on two integral tensor with floating alpha
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_sub_out(tf.ones(sizes), tf.ones(sizes), /*alpha=*/.7, out));
}

TEST_F(OpSubOutTest, BoolInputTensorsFail) {
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});
  Tensor b = tf.make(sizes, /*data=*/{false, true, true, true});

  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_sub_out(a, b, /*alpha=*/1, out));
}

TEST_F(OpSubOutTest, IntOutputWithFloatInputDies) {
  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Float> tff;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends.
  Tensor a = tfi.make(sizes, /*data=*/{2, 4, 3, 3});
  Tensor b = tff.make(sizes, /*data=*/{2, 4, 3, 3});

  // Destination for the sum.
  Tensor out = tfi.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_sub_out(a, b, /*alpha=*/1, out));
}

TEST_F(OpSubOutTest, BoolOutputWithIntegralInput) {
  // add_out() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;
  TensorFactory<ScalarType::Int> tfi;

  const std::vector<int32_t> sizes = {2, 2};

  // Addends.
  Tensor a = tfi.make(sizes, /*data=*/{false, true, true, false});
  Tensor b = tfi.make(sizes, /*data=*/{2, 3, 4, 3});

  // Destination for the sum.
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_sub_out(a, b, /*alpha=*/1, out));
}

TEST_F(OpSubOutTest, MismatchedNonBroadcastableInputShapesDies) {
  TensorFactory<ScalarType::Int> tf;

  // Subtrahend and minuend with different shapes.
  Tensor a = tf.ones(/*sizes=*/{4, 2});
  Tensor b = tf.ones(/*sizes=*/{2, 2});

  // Destination for the subtraction; matches the shape of one of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{8});

  // Performing substraction on two mismatched tensors should cause an assertion
  // and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(context_, op_sub_out(a, b, /*alpha=*/0, out));
}

TEST_F(OpSubOutTest, MismatchedOutputShapesDies) {
  if (SupportedFeatures::get()->output_resize) {
    GTEST_SKIP()
        << "The current kernel supports implicitly resizing output tensor";
  }

  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Subtrahend and minuend with the same shapes.
  Tensor a = tf.ones(sizes);
  Tensor b = tf.ones(sizes);

  // Destination with a different shape.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Performing substraction two tensors into a mismatched output should cause
  // an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(context_, op_sub_out(a, b, /*alpha=*/0, out));
}

TEST_F(OpSubOutTest, BroadcastDimSizeIsOneAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.20342785120010376,
       0.8211539387702942,
       0.12307500839233398,
       0.8268751502037048,
       0.6484894752502441,
       0.8079752326011658});
  Tensor y = tf.make({1, 2}, {0.22279858589172363, 0.3636378049850464});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.019370734691619873,
       0.4575161337852478,
       -0.09972357749938965,
       0.46323734521865845,
       0.4256908893585205,
       0.4443374276161194});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_sub_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSubOutTest, BroadcastDimSizeMissingAB) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.20342785120010376,
       0.8211539387702942,
       0.12307500839233398,
       0.8268751502037048,
       0.6484894752502441,
       0.8079752326011658});
  Tensor y = tf.make({2}, {0.22279858589172363, 0.3636378049850464});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.019370734691619873,
       0.4575161337852478,
       -0.09972357749938965,
       0.46323734521865845,
       0.4256908893585205,
       0.4443374276161194});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_sub_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSubOutTest, BroadcastDimSizeIsOneBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.22279858589172363, 0.3636378049850464});
  Tensor y = tf.make(
      {3, 2},
      {0.20342785120010376,
       0.8211539387702942,
       0.12307500839233398,
       0.8268751502037048,
       0.6484894752502441,
       0.8079752326011658});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.019370734691619873,
       -0.4575161337852478,
       0.09972357749938965,
       -0.46323734521865845,
       -0.4256908893585205,
       -0.4443374276161194});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_sub_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSubOutTest, BroadcastDimSizeMissingBA) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make({1, 2}, {0.22279858589172363, 0.3636378049850464});
  Tensor y = tf.make(
      {3, 2},
      {0.20342785120010376,
       0.8211539387702942,
       0.12307500839233398,
       0.8268751502037048,
       0.6484894752502441,
       0.8079752326011658});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.019370734691619873,
       -0.4575161337852478,
       0.09972357749938965,
       -0.46323734521865845,
       -0.4256908893585205,
       -0.4443374276161194});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_sub_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSubOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.44215160608291626,
       0.17627692222595215,
       0.46265703439712524,
       0.04357701539993286,
       0.838569700717926,
       0.06833052635192871});
  Tensor y = tf.make(
      {3, 2},
      {0.06382524967193604,
       0.18627053499221802,
       0.5863531231880188,
       0.12181782722473145,
       0.5662856698036194,
       0.930520236492157});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.3783263564109802,
       -0.00999361276626587,
       -0.12369608879089355,
       -0.07824081182479858,
       0.27228403091430664,
       -0.8621897101402283});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_sub_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSubOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.44215160608291626,
       0.17627692222595215,
       0.46265703439712524,
       0.04357701539993286,
       0.838569700717926,
       0.06833052635192871});
  Tensor y = tf.make(
      {3, 2},
      {0.06382524967193604,
       0.18627053499221802,
       0.5863531231880188,
       0.12181782722473145,
       0.5662856698036194,
       0.930520236492157});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.3783263564109802,
       -0.00999361276626587,
       -0.12369608879089355,
       -0.07824081182479858,
       0.27228403091430664,
       -0.8621897101402283});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_sub_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSubOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.44215160608291626,
       0.17627692222595215,
       0.46265703439712524,
       0.04357701539993286,
       0.838569700717926,
       0.06833052635192871});
  Tensor y = tf.make(
      {3, 2},
      {0.06382524967193604,
       0.18627053499221802,
       0.5863531231880188,
       0.12181782722473145,
       0.5662856698036194,
       0.930520236492157});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.3783263564109802,
       -0.00999361276626587,
       -0.12369608879089355,
       -0.07824081182479858,
       0.27228403091430664,
       -0.8621897101402283});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_sub_out(x, y, 1, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSubScalarOutTest, SanityCheck) {
  TensorFactory<ScalarType::Int> tf_a;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf_out.zeros(sizes);

  op_sub_scalar_out(tf_a.make(sizes, {1, 2, 4, 8}), 0.5, /*alpha=*/1.5, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf_out.make(sizes, {0.25, 1.25, 3.25, 7.25}));
}

TEST_F(OpSubScalarOutTest, OptimizedSanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor out = tf.zeros(sizes);

  op_sub_scalar_out(
      tf.make(sizes, {6.3, 2.1, 5.6, 8.2}), 1.9, /*alpha=*/2.8, out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, {0.98, -3.22, 0.28, 2.88}));
}

TEST_F(OpSubScalarOutTest, DtypeTest_float16_float_int_float16) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Half> tfHalf;

  exec_aten::Tensor self = tfHalf.ones({2, 2});
  exec_aten::Scalar other = exec_aten::Scalar(-1.0);
  exec_aten::Scalar alpha = exec_aten::Scalar(1);
  exec_aten::Tensor out = tfHalf.zeros({2, 2});
  exec_aten::Tensor out_expected = tfHalf.full({2, 2}, 2.0);
  op_sub_scalar_out(self, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
