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

Tensor& op_pdist_forward_out(const Tensor& input, double p, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::_pdist_forward_outf(context, input, p, out);
}

class OpPdistForwardOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpPdistForwardOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor in = tfFloat.make(
      {4, 5}, {0, 1, 2, 3, 5, 4, 3, 2, -1, 5, 1, 1, -2, 1, 5, 4, 3, 2, -1, 5});
  Tensor out = tfFloat.zeros({6});

  Tensor l0 = tfFloat.make({6}, {3., 3., 3., 4., 0., 4.});
  op_pdist_forward_out(in, 0.0, out);
  EXPECT_TENSOR_CLOSE(out, l0);

  Tensor l0p5 = tfFloat.make(
      {6},
      {29.31370926, 19.48528290, 29.31370926, 43.03986740, 0.0, 43.03986740});
  op_pdist_forward_out(in, 0.5, out);
  EXPECT_TENSOR_CLOSE(out, l0p5);

  Tensor l1 = tfFloat.make({6}, {10., 7., 10., 11., 0., 11.});
  op_pdist_forward_out(in, 1.0, out);
  EXPECT_TENSOR_CLOSE(out, l1);

  Tensor l1p5 = tfFloat.make(
      {6}, {7.07743692, 5.19140196, 7.07743692, 7.08359480, 0.0, 7.08359480});
  op_pdist_forward_out(in, 1.5, out);
  EXPECT_TENSOR_CLOSE(out, l1p5);

  Tensor l2 =
      tfFloat.make({6}, {6.0, 4.58257580, 6.0, 5.74456263, 0.0, 5.74456263});
  op_pdist_forward_out(in, 2.0, out);
  EXPECT_TENSOR_CLOSE(out, l2);

  Tensor l3 = tfFloat.make(
      {6}, {5.14256334, 4.17933941, 5.14256334, 4.74745941, 0.0, 4.74745941});
  op_pdist_forward_out(in, 3.0, out);
  EXPECT_TENSOR_CLOSE(out, l3);

  Tensor linf = tfFloat.make({6}, {4., 4., 4., 4., 0., 4.});
  op_pdist_forward_out(in, INFINITY, out);
  EXPECT_TENSOR_CLOSE(out, linf);
}
