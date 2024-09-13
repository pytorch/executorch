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
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_flip_out(const Tensor& input, IntArrayRef dims, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::flip_outf(context, input, dims, out);
}

class OpFlipOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpFlipOutTest, SmokeTest1Dim) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor input =
      tfFloat.make({4, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  int64_t dims_data[1] = {-1};
  IntArrayRef dims = IntArrayRef(dims_data, 1);
  Tensor out = tfFloat.zeros({4, 1, 3});
  Tensor out_expected =
      tfFloat.make({4, 1, 3}, {3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10});
  op_flip_out(input, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpFlipOutTest, SmokeTest2Dims) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor input =
      tfFloat.make({4, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  int64_t dims_data[2] = {-1, 0};
  IntArrayRef dims = IntArrayRef(dims_data, 2);
  Tensor out = tfFloat.zeros({4, 1, 3});
  Tensor out_expected =
      tfFloat.make({4, 1, 3}, {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  op_flip_out(input, dims, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
