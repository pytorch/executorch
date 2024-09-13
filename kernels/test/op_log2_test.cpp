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
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& op_log2_out(const Tensor& a, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::log2_outf(context, a, out);
}

TEST(OpLog2OutTest, SmokeTest) {
  TensorFactory<ScalarType::Byte> tfByte;
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfByte.make({3}, {45, 55, 82});
  Tensor out = tfFloat.zeros({3});
  Tensor out_expected = tfFloat.make(
      {3}, {5.4918532371521, 5.781359672546387, 6.3575520515441895});
  op_log2_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
