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
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpGluOutTest : public OperatorTest {
 protected:
  Tensor& op_glu_out(const Tensor& self, int64_t dim, Tensor& out) {
    return torch::executor::aten::glu_outf(context_, self, dim, out);
  }

  template <ScalarType DTYPE, ScalarType OUT_DTYPE>
  void expect_tensor_close(Tensor actual, Tensor expected) {
    if (DTYPE == ScalarType::Half || DTYPE == ScalarType::BFloat16 ||
        OUT_DTYPE == ScalarType::Half || OUT_DTYPE == ScalarType::BFloat16) {
      EXPECT_TENSOR_CLOSE_WITH_TOL(
          actual,
          expected,
          1e-2,
          executorch::runtime::testing::internal::kDefaultAtol);
    } else {
      EXPECT_TENSOR_CLOSE(actual, expected);
    }
  }

  // Common testing for glu operator
  template <ScalarType DTYPE, ScalarType OUT_DTYPE>
  void test_glu_out() {
    TensorFactory<DTYPE> tf;
    TensorFactory<OUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {4, 2};
    const std::vector<int32_t> out_sizes_1 = {2, 2};

    // Valid input should give the expected output
    Tensor in = tf.make(sizes, {0, 1, 2, 3, 4, 5, 6, 7});
    Tensor out = tf_out.zeros(out_sizes_1);
    op_glu_out(in, 0, out);
    expect_tensor_close<DTYPE, OUT_DTYPE>(
        out,
        tf_out.make(
            out_sizes_1, /*data=*/{0, 0.99330717, 1.99505484, 2.99726701}));
    const std::vector<int32_t> out_sizes_2 = {4, 1};
    out = tf_out.zeros(out_sizes_2);
    op_glu_out(in, 1, out);
    expect_tensor_close<DTYPE, OUT_DTYPE>(
        out,
        tf_out.make(
            out_sizes_2, /*data=*/{0, 1.90514827, 3.97322869, 5.99453402}));
  }

  // Mismatched shape tests.
  template <ScalarType INPUT_DTYPE>
  void test_glu_out_mismatched_shape() {
    TensorFactory<INPUT_DTYPE> tf_in;

    // Input tensor and out tensor dimension size mismatch
    Tensor in = tf_in.zeros(/*sizes=*/{4, 4, 4});
    Tensor out = tf_in.zeros(/*sizes=*/{2, 4, 2});

    ET_EXPECT_KERNEL_FAILURE(context_, op_glu_out(in, 0, out));

    out = tf_in.zeros(/*sizes=*/{4, 4, 4});
    ET_EXPECT_KERNEL_FAILURE(context_, op_glu_out(in, 0, out));
  }

  // Invalid dimensions tests.
  template <ScalarType INPUT_DTYPE>
  void test_glu_out_invalid_dim() {
    TensorFactory<INPUT_DTYPE> tf_in;
    Tensor in = tf_in.zeros(/*sizes=*/{2, 2});
    const std::vector<int32_t> out_sizes = {1, 2};
    Tensor out = tf_in.zeros(out_sizes);

    // Dim is not valid
    ET_EXPECT_KERNEL_FAILURE(context_, op_glu_out(in, 3, out));

    // Dim size is not even
    in = tf_in.zeros(/*sizes=*/{3, 2});
    ET_EXPECT_KERNEL_FAILURE(context_, op_glu_out(in, 0, out));
  }

  // Unhandled input dtypes.
  template <ScalarType INPUT_DTYPE>
  void test_div_invalid_input_dtype_dies() {
    TensorFactory<INPUT_DTYPE> tf_in;
    TensorFactory<ScalarType::Float> tf_float;

    const std::vector<int32_t> sizes = {2, 2};
    const std::vector<int32_t> out_sizes = {1, 2};
    Tensor in = tf_in.ones(sizes);
    Tensor out = tf_float.zeros(out_sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_glu_out(in, 0, out));
  }

  // Unhandled output dtypes.
  template <ScalarType OUTPUT_DTYPE>
  void test_div_invalid_output_dtype_dies() {
    TensorFactory<ScalarType::Float> tf_float;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 2};
    const std::vector<int32_t> out_sizes = {1, 2};
    Tensor in = tf_float.ones(sizes);
    Tensor out = tf_out.zeros(out_sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_glu_out(in, 0, out));
  }
};

TEST_F(OpGluOutTest, AllInputFloatOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_glu_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, AllInputDoubleOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_glu_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, AllInputHalfOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_glu_out<ScalarType::dtype, ScalarType::Half>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, AllInputBFloat16OutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_glu_out<ScalarType::dtype, ScalarType::BFloat16>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes = {4, 2};
  const std::vector<int32_t> out_sizes = {4, 1};
  Tensor in = tf.make(
      sizes, /*data=*/{INFINITY, 1, -INFINITY, 1, INFINITY, -INFINITY, NAN, 1});
  Tensor out = tf.zeros(out_sizes);
  op_glu_out(in, 1, out);
  EXPECT_TENSOR_CLOSE(
      out,
      tf.make(
          /*sizes=*/out_sizes, /*data=*/{INFINITY, -INFINITY, NAN, NAN}));
}

TEST_F(OpGluOutTest, MismatchedShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_glu_out_mismatched_shape<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, InvalidDimDies) {
#define TEST_ENTRY(ctype, dtype) test_glu_out_invalid_dim<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, AllNonFloatInputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_div_invalid_input_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_div_invalid_output_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpGluOutTest, DynamicShapeUpperBoundSameAsExpected) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {4, 2},
      {0.057747602462768555,
       0.8781633377075195,
       0.4503108263015747,
       0.40363800525665283,
       0.3379024863243103,
       0.13906866312026978,
       0.6991606950759888,
       0.4374786615371704});
  Tensor expected_result = tf.make(
      {2, 2},
      {0.0337061733007431,
       0.4695638120174408,
       0.3008083701133728,
       0.2452739030122757});

  Tensor out =
      tf.zeros({4, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_glu_out(x, 0, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpGluOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {4, 2},
      {0.057747602462768555,
       0.8781633377075195,
       0.4503108263015747,
       0.40363800525665283,
       0.3379024863243103,
       0.13906866312026978,
       0.6991606950759888,
       0.4374786615371704});
  Tensor expected_result = tf.make(
      {2, 2},
      {0.0337061733007431,
       0.4695638120174408,
       0.3008083701133728,
       0.2452739030122757});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_glu_out(x, 0, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpGluOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {4, 2},
      {0.057747602462768555,
       0.8781633377075195,
       0.4503108263015747,
       0.40363800525665283,
       0.3379024863243103,
       0.13906866312026978,
       0.6991606950759888,
       0.4374786615371704});
  Tensor expected_result = tf.make(
      {2, 2},
      {0.0337061733007431,
       0.4695638120174408,
       0.3008083701133728,
       0.2452739030122757});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_glu_out(x, 0, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
