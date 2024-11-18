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

Tensor& op_reflection_pad3d_out(
    const Tensor& input,
    ArrayRef<int64_t> padding,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::reflection_pad3d_outf(
      context, input, padding, out);
}

class OpReflectionPad3DOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpReflectionPad3DOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self =
      tfFloat.make({1, 2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[6] = {1, 1, 2, 1, 1, 0};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 6);
  Tensor out = tfFloat.zeros({1, 3, 6, 4});
  // clang-format off
  Tensor out_expected = tfFloat.make(
    {1, 3, 6, 4},
    {
      11, 10, 11, 10,
       9,  8,  9,  8,
       7,  6,  7,  6,
       9,  8,  9,  8,
      11, 10, 11, 10,
       9,  8,  9,  8,

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
  op_reflection_pad3d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReflectionPad3DOutTest, SmokeTestNegFrontPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self =
      tfFloat.make({1, 2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[6] = {1, 1, 1, -2, -1, 0};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 6);
  Tensor out = tfFloat.zeros({1, 1, 2, 4});
  Tensor out_expected = tfFloat.make({1, 1, 2, 4}, {9, 8, 9, 8, 7, 6, 7, 6});
  op_reflection_pad3d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReflectionPad3DOutTest, SmokeTestNegBackPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self =
      tfFloat.make({1, 2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[6] = {1, 1, 1, 1, 1, -2};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 6);
  Tensor out = tfFloat.zeros({1, 1, 5, 4});
  // clang-format off
  Tensor out_expected = tfFloat.make(
    {1, 1, 5, 4},
    {
       9,  8,  9,  8,
       7,  6,  7,  6,
       9,  8,  9,  8,
      11, 10, 11, 10,
       9,  8,  9,  8
    }
  );
  op_reflection_pad3d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
