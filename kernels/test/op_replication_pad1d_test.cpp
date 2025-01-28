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

Tensor& op_replication_pad1d_out(
    const Tensor& input,
    ArrayRef<int64_t> padding,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::replication_pad1d_outf(
      context, input, padding, out);
}

class OpReplicationPad1DOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpReplicationPad1DOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3}, {0, 1, 2, 3, 4, 5});
  int64_t padding_data[2] = {1, 2};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 2);
  Tensor out = tfFloat.zeros({2, 6});
  Tensor out_expected =
      tfFloat.make({2, 6}, {0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5});
  op_replication_pad1d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReplicationPad1DOutTest, SmokeTestNegLeftPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3}, {0, 1, 2, 3, 4, 5});
  int64_t padding_data[2] = {-1, 1};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 2);
  Tensor out = tfFloat.zeros({2, 3});
  Tensor out_expected = tfFloat.make({2, 3}, {1, 2, 2, 4, 5, 5});
  op_replication_pad1d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpReplicationPad1DOutTest, SmokeTestNegRightPad) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3}, {0, 1, 2, 3, 4, 5});
  int64_t padding_data[2] = {3, -5};
  ArrayRef<int64_t> padding = ArrayRef<int64_t>(padding_data, 2);
  Tensor out = tfFloat.zeros({2, 1});
  Tensor out_expected = tfFloat.make({2, 1}, {0, 3});
  op_replication_pad1d_out(self, padding, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
