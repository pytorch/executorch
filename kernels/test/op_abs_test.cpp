/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpAbsTest : public OperatorTest {
 protected:
  Tensor& op_abs_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::abs_outf(context_, self, out);
  }
};

TEST_F(OpAbsTest, SanityCheck) {
  TensorFactory<ScalarType::Float> tf;

  Tensor in = tf.make({1, 7}, {-3.0, -2.5, -1.01, 0.0, 1.01, 2.5, 3.0});
  Tensor out = tf.zeros({1, 7});
  Tensor expected = tf.make({1, 7}, {3.0, 2.5, 1.01, 0.0, 1.01, 2.5, 3.0});

  Tensor ret = op_abs_out(in, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpAbsTest, MemoryFormatCheck) {
  TensorFactory<ScalarType::Float> tf;

  std::vector<int32_t> sizes = {2, 3, 1, 5};

  Tensor input_contiguous =
      tf.make(sizes, {0.8737,  0.5359,  0.3743,  -0.3040, -0.7800, -0.2306,
                      -0.7684, -0.5364, 0.3478,  -0.3289, 0.0829,  0.2939,
                      -0.8211, 0.8572,  -0.0802, 0.9252,  -0.2093, 0.9013,
                      -0.4197, 0.3987,  -0.5291, -0.5567, 0.2691,  0.7819,
                      -0.8009, -0.4286, -0.9299, 0.2143,  0.2565,  -0.5701});
  Tensor expected_contiguous = tf.make(
      sizes, {0.8737, 0.5359, 0.3743, 0.3040, 0.7800, 0.2306, 0.7684, 0.5364,
              0.3478, 0.3289, 0.0829, 0.2939, 0.8211, 0.8572, 0.0802, 0.9252,
              0.2093, 0.9013, 0.4197, 0.3987, 0.5291, 0.5567, 0.2691, 0.7819,
              0.8009, 0.4286, 0.9299, 0.2143, 0.2565, 0.5701});

  ET_TEST_OP_SUPPORTS_MEMORY_FORMATS(
      tf,
      op_abs_out,
      input_contiguous,
      expected_contiguous,
      /*channels_last_support=*/true);
}
