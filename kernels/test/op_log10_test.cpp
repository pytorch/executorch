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

Tensor& op_log10_out(const Tensor& a, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::log10_outf(context, a, out);
}

TEST(OpLog10OutTest, SmokeTest) {
  TensorFactory<ScalarType::Double> tfDouble;
  TensorFactory<ScalarType::Short> tfShort;

  Tensor self = tfShort.make({8}, {-12, -6, -65, -61, 16, -44, -47, 54});
  Tensor out = tfDouble.zeros({8});
  Tensor out_expected = tfDouble.make(
      {8},
      {NAN, NAN, NAN, NAN, 1.2041200399398804, NAN, NAN, 1.732393741607666});
  op_log10_out(self, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
