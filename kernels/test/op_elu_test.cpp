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

using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::string_view;
using torch::executor::testing::TensorFactory;

class OpEluTest : public OperatorTest {
 protected:
  Tensor& op_elu_out(
      const Tensor& self,
      const Scalar& alpha,
      const Scalar& scale,
      const Scalar& input_scale,
      Tensor& out) {
    return torch::executor::aten::elu_outf(
        context_, self, alpha, scale, input_scale, out);
  }

  template <ScalarType DTYPE>
  void test_elu_execution() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {3, 2};

    Tensor in = tf.make(sizes, /*data=*/{-0.125, -0.25, -1, 0, 1.25, 100});

    Tensor out = tf.zeros(sizes);

    // Run full elu.
    op_elu_out(in, 1.25, 1, 1, out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(
        out,
        tf.make(
            sizes,
            /*data=*/
            {-0.146879, -0.276499, -0.790151, 0, 1.25, 100}));
  }

  template <ScalarType DTYPE>
  void test_integer_elu_dies() {
    TensorFactory<DTYPE> tf;

    Tensor in = tf.ones({1});
    Tensor out = tf.ones({1});
    ET_EXPECT_KERNEL_FAILURE(context_, op_elu_out(in, 1, 1, 1, out));
  }
};

TEST_F(OpEluTest, Basic) {
#define TEST_ENTRY(ctype, dtype) test_elu_execution<ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpEluTest, UnhandledDtypeDies) {
#define TEST_ENTRY(ctype, dtype) test_integer_elu_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpEluTest, MismatchedOutputDtypeDies) {
  // Two different dtypes. This test uses two types with the same size to
  // demonstrate that the ScalarType itself matters, not the size of the
  // tensor elements.
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf_float.ones(sizes);

  // Destination with a dtype different from the input.
  Tensor out = tf_double.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_elu_out(a, 1, 1, 1, out));
}

TEST_F(OpEluTest, MixedScalarTypes) {
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor in = tf_float.ones(sizes);
  Tensor out = tf_float.zeros(sizes);

  op_elu_out(in, true, 1.0, 1.0, out);
  EXPECT_TENSOR_CLOSE(out, tf_float.ones(sizes));

  op_elu_out(in, false, true, 3, out);
  EXPECT_TENSOR_CLOSE(out, tf_float.ones(sizes));
}
