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

Tensor& op_expm1_out(const Tensor& a, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::expm1_outf(context, a, out);
}

TEST(OpExpm1OutTest, SmokeTest) {
  TensorFactory<ScalarType::Double> tfDouble;

  Tensor self = tfDouble.full({}, -31.375);
  Tensor out = tfDouble.zeros({});
  Tensor out_expected = tfDouble.full({}, -0.9999999999999764);
  op_expm1_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
