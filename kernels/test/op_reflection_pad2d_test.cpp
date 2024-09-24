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
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_reflection_pad2d_out(
    const Tensor& input,
    ArrayRef<int64_t> padding,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::reflection_pad2d_outf(
      context, input, padding, out);
}

class OpReflectionPad2DOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpReflectionPad2DOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[4] = {1, 1, 2, 1};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 4);
  Tensor out = tfFloat.zeros({2, 6, 4});
  // clang-format off
  Tensor out_expected = tfFloat.make(
    {2, 6, 4},
    {
      5,  4,  5,  4,
      3,  2,  3,  2,
      1,  0,  1,  0,
      3,  2,  3,  2,
      5,  4,  5,  4,
      3,  2,  3,  2,

      11, 10, 11, 10,
      9,  8,  9,  8,
      7,  6,  7,  6,
      9,  8,  9,  8,
      11, 10, 11, 10,
      9,  8,  9,  8
    }
  );
  // clang-format on
  op_reflection_pad2d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReflectionPad2DOutTest, SmokeTestNegTopPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[4] = {1, 1, -2, 0};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 4);
  Tensor out = tfFloat.zeros({2, 1, 4});
  Tensor out_expected = tfFloat.make({2, 1, 4}, {5, 4, 5, 4, 11, 10, 11, 10});
  op_reflection_pad2d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReflectionPad2DOutTest, SmokeTestNegBottomPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[4] = {1, 1, 1, -3};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 4);
  Tensor out = tfFloat.zeros({2, 1, 4});
  Tensor out_expected = tfFloat.make({2, 1, 4}, {3, 2, 3, 2, 9, 8, 9, 8});
  op_reflection_pad2d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
