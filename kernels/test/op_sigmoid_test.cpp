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
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpSigmoidOutTest : public OperatorTest {
 protected:
  Tensor& op_sigmoid_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::sigmoid_outf(context_, self, out);
  }

  // Common testing for sigmoid operator
  template <ScalarType DTYPE, ScalarType OUTPUT_DTYPE>
  void test_integer_sigmoid_out() {
    TensorFactory<DTYPE> tf;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 2};

    // Destination for the sigmoid operator.
    Tensor out = tf_out.zeros(sizes);

    op_sigmoid_out(tf.make(sizes, /*data=*/{1, 2, 4, 8}), out);

    // Check that it matches (or close to) the expected output.
    EXPECT_TENSOR_CLOSE(
        out,
        tf_out.make(sizes, /*data=*/{0.731059, 0.880797, 0.982014, 0.999665}));
  }

  // Unhandled output dtypes.
  template <ScalarType OUTPUT_DTYPE>
  void test_sigmoid_invalid_output_dtype_dies() {
    TensorFactory<ScalarType::Float> tf;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    Tensor in = tf.ones(sizes);
    Tensor out = tf_out.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_sigmoid_out(in, out));
  }
};

TEST_F(OpSigmoidOutTest, AllRealInputHalfOutputSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_integer_sigmoid_out<ScalarType::dtype, ScalarType::Half>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpSigmoidOutTest, AllRealInputFloatOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_integer_sigmoid_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpSigmoidOutTest, AllRealInputDoubleOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_integer_sigmoid_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Mismatched shape tests.
TEST_F(OpSigmoidOutTest, MismatchedShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Float> tf_out;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf_out.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_sigmoid_out(a, out));
}

TEST_F(OpSigmoidOutTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_sigmoid_invalid_output_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
