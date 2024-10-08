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

Tensor& op_replication_pad2d_out(
    const Tensor& input,
    ArrayRef<int64_t> padding,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::replication_pad2d_outf(
      context, input, padding, out);
}

class OpReplicationPad2DOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpReplicationPad2DOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[4] = {1, 2, 2, 1};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 4);
  Tensor out = tfFloat.zeros({2, 6, 5});
  // clang-format off
  Tensor out_expected = tfFloat.make(
    {2, 6, 5},
    {
      0.,  0.,  1.,  1.,  1.,
      0.,  0.,  1.,  1.,  1.,
      0.,  0.,  1.,  1.,  1.,
      2.,  2.,  3.,  3.,  3.,
      4.,  4.,  5.,  5.,  5.,
      4.,  4.,  5.,  5.,  5.,

      6.,  6.,  7.,  7.,  7.,
      6.,  6.,  7.,  7.,  7.,
      6.,  6.,  7.,  7.,  7.,
      8.,  8.,  9.,  9.,  9.,
      10., 10., 11., 11., 11.,
      10., 10., 11., 11., 11.
    }
  );
  // clang-format on
  op_replication_pad2d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReplicationPad2DOutTest, SmokeTestNegTopPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[4] = {1, 1, -2, 0};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 4);
  Tensor out = tfFloat.zeros({2, 1, 4});
  Tensor out_expected = tfFloat.make({2, 1, 4}, {4, 4, 5, 5, 10, 10, 11, 11});
  op_replication_pad2d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReplicationPad2DOutTest, SmokeTestNegBottomPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  int64_t padding_data[4] = {1, 1, 3, -5};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 4);
  Tensor out = tfFloat.zeros({2, 1, 4});
  Tensor out_expected = tfFloat.make({2, 1, 4}, {0, 0, 1, 1, 6, 6, 7, 7});
  op_replication_pad2d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
