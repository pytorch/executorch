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
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& op_maximum_out(const Tensor& self, const Tensor& other, Tensor& out) {
  executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext context{};
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

TEST(OpMaximumOutTest, SmokeTestLarger) {
  TensorFactory<ScalarType::Float> tfFloat;

  std::vector<float> a(18);
  std::iota(a.begin(), a.end(), -8);
  Tensor self = tfFloat.make({18}, a);
  Tensor other = tfFloat.full({18}, 4);
  Tensor out = tfFloat.zeros({18});
  Tensor out_expected = tfFloat.make(
      {18}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9});
  op_maximum_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
