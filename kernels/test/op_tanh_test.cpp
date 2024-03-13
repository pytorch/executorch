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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpTanhOutTest : public OperatorTest {
 protected:
  Tensor& op_tanh_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::tanh_outf(context_, self, out);
  }

  // Common testing for tanh operator and all kinds of supported input types
  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_floating_point_tanh_out() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {1, 12};

    // Destination for the tanh operator.
    Tensor out = tf_out.zeros(sizes);

    // clang-format off
    op_tanh_out(
        tf_in.make(sizes, /*data=*/{ 0,  1,  2,  3,   4,  5,
                                     6,  7,  8,  9,  10,  100}),
        out);
  
    // Check that it matches (or close to) the expected output.
    EXPECT_TENSOR_CLOSE(
        out,
        tf_out.make(
            sizes, /*data=*/{ 0.0000000000,  0.7615941763,
                               0.9640275836,  0.9950547814,  0.9993293285,
                               0.9999092221,  0.9999877214,  0.9999983311,
                               0.9999997616,  0.9999999404,  1.0000000000, 1.0000000000}));
    // clang-format on
  }

  // Unhandled output dtypes.
  template <ScalarType INPUT_DTYPE, ScalarType OUTPUT_DTYPE>
  void test_tanh_invalid_output_dtype_dies() {
    TensorFactory<INPUT_DTYPE> tf;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    Tensor in = tf.ones(sizes);
    Tensor out = tf_out.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_tanh_out(in, out));
  }
};

TEST_F(OpTanhOutTest, HandleBoolInput) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{false, true});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{0.000000, 0.761594});

  EXPECT_TENSOR_CLOSE(op_tanh_out(a, out), res);
}

TEST_F(OpTanhOutTest, AllRealInputHalfOutputSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_tanh_out<ScalarType::dtype, ScalarType::Half>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpTanhOutTest, AllRealInputFloatOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_tanh_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpTanhOutTest, AllRealInputDoubleOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_tanh_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpTanhOutTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_tanh_invalid_output_dtype_dies<ScalarType::Float, ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Mismatched shape tests.
TEST_F(OpTanhOutTest, MismatchedInputShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched input shapes";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_tanh_out(a, out));
}

TEST_F(OpTanhOutTest, SimpleGeneratedCase) {
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
  Tensor expected_result = tf.make(
      {10, 10}, {0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194, 0.7615941762924194, 0.7615941762924194,
                 0.7615941762924194});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_tanh_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpTanhOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.23026639223098755,
       0.24356824159622192,
       0.9074369668960571,
       0.167863667011261,
       0.8099868297576904,
       0.6270960569381714});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.22628112137317657,
       0.2388632595539093,
       0.7198998332023621,
       0.1663045436143875,
       0.6695830225944519,
       0.5560494065284729});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_tanh_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpTanhOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.23026639223098755,
       0.24356824159622192,
       0.9074369668960571,
       0.167863667011261,
       0.8099868297576904,
       0.6270960569381714});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.22628112137317657,
       0.2388632595539093,
       0.7198998332023621,
       0.1663045436143875,
       0.6695830225944519,
       0.5560494065284729});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_tanh_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpTanhOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.23026639223098755,
       0.24356824159622192,
       0.9074369668960571,
       0.167863667011261,
       0.8099868297576904,
       0.6270960569381714});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.22628112137317657,
       0.2388632595539093,
       0.7198998332023621,
       0.1663045436143875,
       0.6695830225944519,
       0.5560494065284729});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_tanh_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
