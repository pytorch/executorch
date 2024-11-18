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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_maximum_out(const Tensor& self, const Tensor& other, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::maximum_outf(context, self, other, out);
}

TEST(OpMaximumOutTest, SmokeTest) {
  TensorFactory<ScalarType::Double> tfDouble;
  TensorFactory<ScalarType::Float> tfFloat;
  TensorFactory<ScalarType::Short> tfShort;

  Tensor self = tfFloat.full({}, -86.125);
  Tensor other = tfShort.full({}, 36);
  Tensor out = tfDouble.zeros({});
  Tensor out_expected = tfDouble.full({}, 36.0);
  op_maximum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
