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

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpRepeatInterleaveTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_repeat_out(
      const Tensor& repeats,
      optional<int64_t> output_size,
      Tensor& out) {
    return torch::executor::aten::repeat_interleave_outf(
        context_, repeats, output_size, out);
  }
};

TEST_F(OpRepeatInterleaveTensorOutTest, SmokeTest) {
  TensorFactory<ScalarType::Int> tf;

  Tensor repeats = tf.make({3}, {2, 3, 1});

  std::vector<int64_t> repeats_vec = {3, 4, 5, 6};
  Tensor out = tf.zeros({6});
  Tensor expected = tf.make({6}, {0, 0, 1, 1, 1, 2});
  Tensor ret = op_repeat_out(repeats, 6, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}
