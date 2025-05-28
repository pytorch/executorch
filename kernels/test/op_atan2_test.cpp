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
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& op_atan2_out(const Tensor& self, const Tensor& other, Tensor& out) {
  executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext context{};
  return torch::executor::aten::atan2_outf(context, self, other, out);
}

TEST(OpAtan2OutTest, SmokeTest) {
  TensorFactory<ScalarType::Double> tfDouble;
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self =
      tfDouble.make({3, 2}, {20.25, 42.5, 51.625, -46.125, 80.375, -35.75});
  Tensor other = tfDouble.make({2}, {-0.625, -2.25});
  Tensor out = tfFloat.zeros({3, 2});
  Tensor out_expected = tfFloat.make(
      {3, 2},
      {1.6016507148742676,
       1.6236881017684937,
       1.5829023122787476,
       -1.6195381879806519,
       1.5785722732543945,
       -1.633650541305542});
  op_atan2_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST(OpAtan2OutTest, SmokeTestNoBroadcastingSameDtype) {
  TensorFactory<ScalarType::Double> tfDouble;

  std::vector<double> a(18);
  std::iota(a.begin(), a.end(), -8);
  std::vector<double> b(18, 2.0);
  Tensor self = tfDouble.make({18}, a);
  Tensor other = tfDouble.make({18}, b);
  Tensor out = tfDouble.zeros({18});
  Tensor out_expected = tfDouble.make(
      {18},
      {-1.3258176636680326,
       -1.2924966677897853,
       -1.2490457723982544,
       -1.1902899496825317,
       -1.1071487177940904,
       -0.9827937232473291,
       -0.7853981633974483,
       -0.4636476090008061,
       0.0000000000000000,
       0.4636476090008061,
       0.7853981633974483,
       0.9827937232473291,
       1.1071487177940904,
       1.1902899496825317,
       1.2490457723982544,
       1.2924966677897853,
       1.3258176636680326,
       1.3521273809209546});
  op_atan2_out(self, other, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
